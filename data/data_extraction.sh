array=(  100050716 100095342 100099673 100119147 100119331 100167523 100176564 100186518 100191721 100232826 100245868 100283482 100302025 100333714 100346741 100347538 100443358 100475925 100534615 100585216 100631293 100648742 100658683 100669789 100705014 100761192 100767852 100774542 100786280 100815191 100824468 100832916 100841439 100862002 100863944 100909042 100926139 100928249 100937795 100945650 100946606 100950405 100963093 100985962 101037004 101055065 101058733 101092542 101204936 101271343 101310439 101316371 101331604 101370154 101405067 101421001 101424740 101429446 101461933 101473375 101478878 101557560 101559810 101561266 101570167 101635841 101676130 101681957 101687203 101716429 101748084 101752035 101781258 101799974 101830913 101858430 101869708 101894494 101895103 101931782 102004691 102027839 102032127 102032735 102090373 102090426 102093227 102103873 102112619 102165114 102209009 102246367 102251694 102266652 102286240 102335870 102348301 102414649 102503835 102516908 102617016 102665177 102756174 102761280 102785451 102788479 102804150 102851974 102861910 102865564 102871703 102878697 102894800 102895043 102898975 102911771 102935647 102949794 102965473 102970373 102970521 103049951 103120121 103253901 103375468 103483570 103514617 103540418 103541423 103569864 103578775 103583711 103593515 103654741 103663397 103684076 103694080 103739079 103739467 103781141 103797865 103801263 103811377 103840360 103843159 103867717 103890381 103913418 103937198 103937874 103959690 103984893 104030233 104070198 104146882 104154009 104184027 104187176 104189945 104197217 104201622 104217338 104218196 104220125 104286765 104313006 104344148 104358881 104372191 104389752 104404042 104406514 104443919 104450226 104464144 104579719 104587309 104624379 104627092 104635033 104637661 104666711 104675153 104681363 104682681 104687983 104759733 104765714 104788974 104793371 104793667 104803013 104815041 104817210 104847207 104932446 104958496 105011977 105091965 105097600 )

mkdir mini_data_folder

for i in "${array[@]}"
do
	cp -r $i mini_data_folder/
done

zip -r mini.zip mini_data_folder