import json
import tensorflow as tf


"""
DataGenerator file to handle 
"""
class OutfitSequencer(tf.keras.utils.Sequence):
    def __init__(self,
                 batch_size=10,
                 image_dim= 224,
                 root_dir="/content/drive/MyDrive/ML_Final_Project/data/images/",
                 data_file="/content/drive/MyDrive/ML_Final_Project/data/train_no_dup_with_category_3more_name.json",
                 data_dir="/content/drive/MyDrive/ML_Final_Project/data/",
                 use_mean_img=True,
                 neg_samples=True):
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.use_mean_img = use_mean_img

        # extracting and storing training samples as a list
        self.data = json.load(open(data_file))
        self.data = [(k, v) for k, v in self.data.items()]

        self.neg_samples = neg_samples # if True, will randomly generate negative outfit samples

        # datalength
        self.dataLen = len(self.data)


        self.vocabulary, self.word_to_idx = [], {}
        self.word_to_idx['UNK'] = len(self.word_to_idx)
        self.vocabulary.append('UNK')
        with open(data_dir+'final_word_dict.txt') as f:
            for line in f:
                name = line.strip().split()[0]
                if name not in self.word_to_idx:
                    self.word_to_idx[name] = len(self.word_to_idx)
                    self.vocabulary.append(name)


    def transform(self, image, dim):
        # resize
        tf.image.resize_images(image, [dim, dim])
        return image


    def image_loader(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        # converting it tensor
        image = tf.cast(image, tf.float32)
        image = tf.expand_dims(image, 0)
        return image
    
    def on_epoch_end(self):
        # Updates indexes after each epoch
        seed = np.random.randint()
        self.data = np.random.shuffle(self.data, random_state=seed)


    def run_for_each_item(self, set_id, parts, index):
        # random choose negative items
        if random.randint(0, 1) and self.neg_samples:
            to_change = list(parts.keys())
        else:
            to_change = []
        imgs = []
        labels = []
        names = []
        for part in ['upper', 'bottom', 'shoe', 'bag', 'accessory']:
            # random choose a image from dataset with same category
            if part in to_change:
                choice = self.data[index]
                while (choice[0] == set_id) or (part not in choice[1].keys()):
                    choice = random.choice(self.data)
                img_path = os.path.join(self.root_dir, str(choice[0]), str(choice[1][part]['index'])+'.jpg')
                names.append(tf.constant(self.str_to_idx(choice[1][part]['name']), dtype=tf.float64))
                labels.append('{}_{}'.format(choice[0], choice[1][part]['index']))
            elif part in parts.keys():
                img_path = os.path.join(self.root_dir, str(set_id), str(parts[part]['index'])+'.jpg')
                names.append(tf.constant(self.str_to_idx(choice[1][part]['name']), dtype=tf.float64))
                labels.append('{}_{}'.format(set_id, parts[part]['index']))
            elif self.use_mean_img:
                # mean_img encoding
                img_path = os.path.join(self.data_dir, part+'.png')
                names.append(tf.constant([], dtype=tf.float64))
                labels.append('{}_{}'.format(part, 'mean'))
            else:
                continue
            img = self.image_loader(img_path)
            imgs.append(img)
        input_images = tf.concat(imgs, 0)
        is_compat = (len(to_change)==0)

        offsets = list(itertools.accumulate([0] + [len(n) for n in names[:-1]]))
        offsets = tf.constant(offsets, dtype=tf.float64)
        return input_images, names, offsets, set_id, labels, is_compat      

    
    def __getitem__(self, x):
        """It could return a positive suits or negative suits"""
        batch = []

        indices = self.data[np.arange(x, x+self.batch_size)]

        for index in indices:
            set_id, parts = self.data[index]
            batch.append(self.run_for_each_item(set_id, parts, index))

        return tf.convert_to_tensor(batch)

    
    def str_to_idx(self, name):
        return [self.word_to_idx[w] if w in self.word_to_idx else self.word_to_idx['UNK']
            for w in name.split()]
    
    def __len__(self):
        return self.dataLen // self.batch_size