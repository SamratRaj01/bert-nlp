# 20 paired clean/noisy records

| example_id | operation | intensity | med | normalized_med | clean_token_count | noisy_token_count | token_count_change | sequence_change_rate | observation |
|---|---|---|---|---|---|---|---|---|---|
| wt2-train-000004 | insertion | 0.000 | 0 | 0.000 | 105 | 105 | 0 | 0.000 | intensity-0 control: no edits, tokenization identical. |
| wt2-train-000005 | insertion | 0.050 | 25 | 0.044 | 110 | 150 | 40 | 0.527 | 25 edit(s), token count +40, continuation pieces +40. |
| wt2-train-000011 | deletion | 0.100 | 50 | 0.101 | 87 | 121 | 34 | 0.816 | 50 edit(s), token count +34, continuation pieces +41. |
| wt2-train-000031 | substitution | 0.150 | 70 | 0.142 | 98 | 186 | 88 | 1.469 | 70 edit(s), token count +88, continuation pieces +99. |
| wt2-train-000035 | insertion | 0.050 | 13 | 0.042 | 76 | 96 | 20 | 0.382 | 13 edit(s), token count +20, continuation pieces +19. |
| wt2-train-000038 | deletion | 0.100 | 53 | 0.107 | 104 | 131 | 27 | 0.692 | 53 edit(s), token count +27, continuation pieces +40. |
| wt2-train-000042 | substitution | 0.150 | 42 | 0.113 | 77 | 122 | 45 | 0.974 | 42 edit(s), token count +45, continuation pieces +52. |
| wt2-train-000053 | insertion | 0.050 | 28 | 0.051 | 109 | 156 | 47 | 0.642 | 28 edit(s), token count +47, continuation pieces +47. |
| wt2-train-000054 | deletion | 0.100 | 43 | 0.086 | 93 | 127 | 34 | 0.753 | 43 edit(s), token count +34, continuation pieces +41. |
| wt2-train-000063 | substitution | 0.150 | 68 | 0.138 | 94 | 154 | 60 | 1.160 | 68 edit(s), token count +60, continuation pieces +72. |
| wt2-train-000064 | insertion | 0.050 | 4 | 0.093 | 12 | 18 | 6 | 0.917 | 4 edit(s), token count +6, continuation pieces +5. |
| wt2-train-000067 | deletion | 0.100 | 29 | 0.111 | 45 | 71 | 26 | 1.044 | 29 edit(s), token count +26, continuation pieces +31. |
| wt2-train-000068 | substitution | 0.150 | 16 | 0.190 | 16 | 34 | 18 | 1.750 | 16 edit(s), token count +18, continuation pieces +21. |
| wt2-train-000069 | insertion | 0.050 | 7 | 0.055 | 21 | 32 | 11 | 0.714 | 7 edit(s), token count +11, continuation pieces +11. |
| wt2-train-000070 | deletion | 0.100 | 11 | 0.093 | 22 | 27 | 5 | 0.591 | 11 edit(s), token count +5, continuation pieces +9. |
| wt2-train-000071 | substitution | 0.150 | 60 | 0.155 | 78 | 143 | 65 | 1.359 | 60 edit(s), token count +65, continuation pieces +73. |
| wt2-train-000072 | insertion | 0.050 | 30 | 0.063 | 86 | 135 | 49 | 0.802 | 30 edit(s), token count +49, continuation pieces +49. |
| wt2-train-000073 | deletion | 0.100 | 36 | 0.120 | 69 | 84 | 15 | 0.696 | 36 edit(s), token count +15, continuation pieces +22. |
| wt2-train-000075 | substitution | 0.150 | 9 | 0.200 | 14 | 20 | 6 | 0.786 | 9 edit(s), token count +6, continuation pieces +7. |
| wt2-train-000076 | insertion | 0.050 | 0 | 0.000 | 13 | 13 | 0 | 0.000 | intensity 0.05: no character positions were sampled for this short text, tokenization identical. |

## Full text per pair

### wt2-train-000004  (insertion, intensity 0.0)
- clean: `The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more forgiving for series newcomers . Character designer Raita Honjou and composer Hitoshi Sakimoto both returned from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large team of writers handled the script . The game 's opening theme was sung by May 'n .`
- noisy: `The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more forgiving for series newcomers . Character designer Raita Honjou and composer Hitoshi Sakimoto both returned from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large team of writers handled the script . The game 's opening theme was sung by May 'n .`
- MED 0 (normalized 0.000); tokens 105 -> 105 (change +0)
- intensity-0 control: no edits, tokenization identical.

### wt2-train-000005  (insertion, intensity 0.05)
- clean: `It met with positive sales in Japan , and was praised by both Japanese and western critics . After release , it received downloadable content , along with an expanded edition in November of that year . It was also adapted into manga and an original video animation series . Due to low sales of Valkyria Chronicles II , Valkyria Chronicles III was not localized , but a fan translation compatible with the game 's expanded edition was released in 2014 . Media.Vision would return to the franchise with the development of Valkyria : Azure Revolution for the PlayStation 4 .`
- noisy: `It met with positive sales in Japan , and was praivsed by both Jfapanese and wemstern critics . After renleakse , uit received downloadable content , along with an expanded edition in November of that year . It was alsop adapted igncto emanga and an original videox animation series . Due to low sales onf Valkyria Chronicles IIs , Valkyria Chrongicles III was ngot localized , but a fan translation compatible with the game 's expanded edition was relqeased in 2014 . Media.Vision would raeturn to dthe franchiesxe wilth the bdevelopment of Valkyryia : Azure Revolustion for the PlayStationv 4 .`
- MED 25 (normalized 0.044); tokens 110 -> 150 (change +40)
- 25 edit(s), token count +40, continuation pieces +40.

### wt2-train-000011  (deletion, intensity 0.1)
- clean: `Troops are divided into five classes : Scouts , Shocktroopers , Engineers , Lancers and Armored Soldier . Troopers can switch classes by changing their assigned weapon . Changing class does not greatly affect the stats gained while in a previous class . With victory in battle , experience points are awarded to the squad , which are distributed into five different attributes shared by the entire squad , a feature differing from early games ' method of distributing to different unit types .`
- noisy: `Troos are divide ino ive classes: Scouts ,Shoktroopers , Engneers , Lancer and Armored Soldir . Troperscan swich cases by canging their assiged weapon . Caging class does not greatly affect te stats gaine while in a preious clas .Wth victoryin battle  xerienc pointsare awarded t hesquad , which are distributed into five differentatributs shared by the enre quad ,a featediffering from early gmes ' method f istributng to difeent unit types .`
- MED 50 (normalized 0.101); tokens 87 -> 121 (change +34)
- 50 edit(s), token count +34, continuation pieces +41.

### wt2-train-000031  (substitution, intensity 0.15)
- clean: `Unlike its two predecessors , Valkyria Chronicles III was not released in the west . According to Sega , this was due to poor sales of Valkyria Chronicles II and the general unpopularity of the PSP in the west . An unofficial fan translation patch began development in February 2012 : players with a copy of Valkyria Chronicles III could download and apply the patch , which translated the game 's text into English . Compatible with the Extra Edition , the patch was released in January 2014 .`
- noisy: `Unlike ats two pnedecesiors , Vtlkyriajbhronicles IIIgwas nok relecmed in ths qest . Aocooding tohSega d this wfs due to poor sajes of Vilkyria khroniclesyIItand the gentral unpmpularity ofjthe PSP ik tht westu.aAn unosfjcial fan trakslationlpntcowbeganadeceaoymeqt in Febroary 2ut2 : playeks witt a copy of Vwlkyria uhroniwles IsI douldmaownload and apply the patch , wmich translatzd the gamx 's text inoo Enclish .oCxmpatible with the Exncf Editioy a the paych was releised iu Januarke2014 .`
- MED 70 (normalized 0.142); tokens 98 -> 186 (change +88)
- 70 edit(s), token count +88, continuation pieces +99.

### wt2-train-000035  (insertion, intensity 0.05)
- clean: `On its day of release in Japan , Valkyria Chronicles III topped both platform @-@ exclusive and multi @-@ platform sales charts . By early February , the game sold 102 @,@ 779 units , coming in second overall to The Last Story for the Wii . By the end of the year , the game had sold just over 152 @,@ 500 units .`
- noisy: `On imts day of rcelease in Jappan , Valkyria Chroniclses III topped both platform @-@ wexclusivze and multi @-@g platform sales charts . By early February , the game sold 102 @,@ 779 units , coming in second overaall tmo The Lastd Story for the Wii . By the end of the yearc , the game had sold just vover 152 @,@ 500 unitds .`
- MED 13 (normalized 0.042); tokens 76 -> 96 (change +20)
- 13 edit(s), token count +20, continuation pieces +19.

### wt2-train-000038  (deletion, intensity 0.1)
- clean: `In a preview of the TGS demo , Ryan Geddes of IGN was left excited as to where the game would go after completing the demo , along with enjoying the improved visuals over Valkyria Chronicles II . Kotaku 's Richard Eisenbeis was highly positive about the game , citing is story as a return to form after Valkyria Chronicles II and its gameplay being the best in the series . His main criticisms were its length and gameplay repetition , along with expressing regret that it would not be localized .`
- noisy: `In a previewof theTGSdemo , Ran Geddes of IGN was eft ectedas to here the gmewouldgo after cmplting the deo , long withenjongthe improved visuals over alkyria Chrocles I . Kotaku 's Richard iseneis was highly positive aboutthe ame , ctinis tory as a return o form atr Valkyria Chronicle II and itsgameplay being the best in the series .His main critiiss wer itslent and gamelay repetiion , aong wthexpressing regret tht it woul no be localizd .`
- MED 53 (normalized 0.107); tokens 104 -> 131 (change +27)
- 53 edit(s), token count +27, continuation pieces +40.

### wt2-train-000042  (substitution, intensity 0.15)
- clean: `Kurt and Riela were featured in the Nintendo 3DS crossover Project X Zone , representing the Valkyria series . Media.Vision would return to the series to develop Valkyria : Azure Revolution , with Ozawa returning as director . Azure Revolution is a role @-@ playing video game for the PlayStation 4 that forms the beginning of a new series within the Valkyria franchise .`
- noisy: `Kurt anv Rielagtereqfmahured in the Nintendi 3DS crossoveg Projeco X noie , rfpresentsng thetValkyria series . Media.Vision would returndto the srries to segelop ValqpriawyuAzuvx Revolution , witc Ozawa yeturnaeg as director . Azury Revolutiun is a role @-@ playing videg gamedfor tke PlayStftion f that xorms the beginning of a new seriev wytjin the Vapkyriv franchise .`
- MED 42 (normalized 0.113); tokens 77 -> 122 (change +45)
- 42 edit(s), token count +45, continuation pieces +52.

### wt2-train-000053  (insertion, intensity 0.05)
- clean: `The Tower Building of the Little Rock Arsenal , also known as U.S. Arsenal Building , is a building located in MacArthur Park in downtown Little Rock , Arkansas . Built in 1840 , it was part of Little Rock 's first military installation . Since its decommissioning , The Tower Building has housed two museums . It was home to the Arkansas Museum of Natural History and Antiquities from 1942 to 1997 and the MacArthur Museum of Arkansas Military History since 2001 . It has also been the headquarters of the Little Rock Æsthetic Club since 1894 .`
- noisy: `The Tozwer Buildingc of the Little Rock Arsenal , also known as U.Sw. Alrsenal Building , is a building locatewd iwn MacArthrur Park in downitown Lithtle Rock , Arkaensas . Buijlt in 18f40 , it was part of Little Rock 's first military installation . Since its decommissioning , The dTower Bxuxilding has housed two museums . It woaos home toa the Arkansas Museum of Natural kHistory and Antiquitiwes from 1942 to 1997 apnd the MacArthur Museum of Arkansas Military History wsince 20u01 . It has also been the headquarters of trhe Lzisttle Rock Æsthetic Clubs since 1v894 .`
- MED 28 (normalized 0.051); tokens 109 -> 156 (change +47)
- 28 edit(s), token count +47, continuation pieces +47.

### wt2-train-000054  (deletion, intensity 0.1)
- clean: `The building receives its name from its distinct octagonal tower . Besides being the last remaining structure of the original Little Rock Arsenal and one of the oldest buildings in central Arkansas , it was also the birthplace of General Douglas MacArthur , who became the supreme commander of US forces in the South Pacific during World War II . It was also the starting place of the Camden Expedition . In 2011 it was named as one of the top 10 attractions in the state of Arkansas by Arkansas.com.`
- noisy: `The building reeive it name from its disinct octagona toer .Besies being the lat rmaiingstructure of the oriinal Little Rock rsenal an one of the oldest buildings in ental Akansas , t was also the birthplace of Geneal ouglas Marthur , wobecame the supreme commander of US forces in the South Pacifc during WorldWar II . It was also the starting plcef the Camdn Expedition . In 201iwas named s one of te top 10 atractionsin the state of Arkanas y Arkanas.cm.`
- MED 43 (normalized 0.086); tokens 93 -> 127 (change +34)
- 43 edit(s), token count +34, continuation pieces +41.

### wt2-train-000063  (substitution, intensity 0.15)
- clean: `The United States troops at the outposts of the western frontier of the state and in the Indian nation have all been recalled from winter quarters to reinforce the garrison at Fort Smith . The garrison at Fort Smith had been previously transferred to the United States Arsenal in this city ( Little Rock ) . The arsenal is one of the richest depositories of military stores in the United States and is supposed to be the ultimate destination of the tropps [ sic ] ordered from the frontier .`
- noisy: `Tta UnidzdlStates troopz at the outmouts of the wuaaern frontier bf the strtv and ic theaIndiab gation hate all been rxcalled frymywtrter buaroers to feinforce mhe garriqvn at ForpnSmitl . The garrison bt Fort Smith had beet previocsay transferred tofthelUnijed Statys aqyenal inethisqcity f Little Rock ) t The arsenal bs onh of thurrichpst depositories ofpmilitary stires dn the Unitea Statss and ls supposedbtv fe the ultimade destination of the tropps vjsic ] ordered from the froatxec .`
- MED 68 (normalized 0.138); tokens 94 -> 154 (change +60)
- 68 edit(s), token count +60, continuation pieces +72.

### wt2-train-000064  (insertion, intensity 0.05)
- clean: `-John M Harrel Telegram , January 31 , 1861`
- noisy: `-John M Hmarrel Telegraem , tJanuary 31 ,j 1861`
- MED 4 (normalized 0.093); tokens 12 -> 18 (change +6)
- 4 edit(s), token count +6, continuation pieces +5.

### wt2-train-000067  (deletion, intensity 0.1)
- clean: `Perhaps because Abraham Lincoln had not yet been inaugurated as President , Captain Totten received no instructions from his superiors and was forced to withdraw his troops . He agreed to surrender the arsenal as long as the governor agreed to three provisions :`
- noisy: `Perhap because Abraam Lincolnhad no e been ingurated asPresident , Captain Tottenreeived no instruction from his supeiors and was foce t withdraw i troops. Heagreed to srender the arsnal aslng s the gvernor agreed to hree povisions :`
- MED 29 (normalized 0.111); tokens 45 -> 71 (change +26)
- 29 edit(s), token count +26, continuation pieces +31.

### wt2-train-000068  (substitution, intensity 0.15)
- clean: `The governor would take possession of the arsenal in the name of the United States .`
- noisy: `ihe ioverqor roxlretake possedsion of the vrsenal in the namejoo fhe UnvtndbStatex .`
- MED 16 (normalized 0.190); tokens 16 -> 34 (change +18)
- 16 edit(s), token count +18, continuation pieces +21.

### wt2-train-000069  (insertion, intensity 0.05)
- clean: `The soldiers would be allowed safe passage in any direction carrying any personal and public property besides munitions of war .`
- noisy: `The soldiers would be allowed safe pasksage in any directionf carrying eany personal and public prcopertyw besides munigtioins of war .`
- MED 7 (normalized 0.055); tokens 21 -> 32 (change +11)
- 7 edit(s), token count +11, continuation pieces +11.

### wt2-train-000070  (deletion, intensity 0.1)
- clean: `The soldiers would be allowed to march away as men leaving under orders , not as conquered and surrendering soldiers .`
- noisy: `The soldiers ouldbeallowed to march away ase leang udr orders , not as conquered and surrenderingsoldiers .`
- MED 11 (normalized 0.093); tokens 22 -> 27 (change +5)
- 11 edit(s), token count +5, continuation pieces +9.

### wt2-train-000071  (substitution, intensity 0.15)
- clean: `On the morning of February 8 , 1861 , Rector and Totten signed an agreement placing the arsenal in the hands of state officials . That afternoon , the citizen militia marched to the arsenal with Governor Rector at its head . All of the federal troops had left at this point , except Totten who had stayed behind to listen to the Governor 's speech and to hand the arsenal over in person .`
- noisy: `Oofthe mtrning ly Febrzaryr8 n 1861 , Rector and Toaten dignes an anreemqnt plating thz arsenrlbrn the hqnms of state oeficijls . That astertoon , the citizen militia marched to the arcldal mjnh Govercor Rbctjrqat uts hswdg. Allqof tkh kederal troops hal left ftwthis point , exeepv Totteh wbo htd stdyed behind to lfsteg to the Governor 's speech and td handjtbe qrsenalmover in persqnq.`
- MED 60 (normalized 0.155); tokens 78 -> 143 (change +65)
- 60 edit(s), token count +65, continuation pieces +73.

### wt2-train-000072  (insertion, intensity 0.05)
- clean: `The Little Rock Arsenal was classified in 1860 as an " arsenal of deposit , " meaning that it was simply a warehouse for the storage of weapons intended for the use of the state militia in times of crisis . Thus there were no substantial operations for ordnance fabrication or repairs , nor for the manufacture of cartridges at the time the Arsenal fell into State hands . Most of these operations were started from scratch through the efforts of the Arkansas Military Board .`
- noisy: `The Littlqe Rock oArsenal dwas xclassified in 1860 as an " arsenal of xdeposit , " meaning thcat it was simply a warehouse for the storage of weapwons ibntended forq the use of the sttate militia in timesn ofc crisis . Thus therze were no substantial operations for hordpnance fabricatlion orx drepairs , nor for thes manufacture of cartridges at the timye tdhe Arsenal fell into State hands . fMoqst tof thesbe operwaltions were started from scratch throught the efforts of othep Arkansas Military Board .`
- MED 30 (normalized 0.063); tokens 86 -> 135 (change +49)
- 30 edit(s), token count +49, continuation pieces +49.

### wt2-train-000073  (deletion, intensity 0.1)
- clean: `Inside the Little Rock Arsenal after its seizure in February , 1861 , the Confederates inventoried some 10 @,@ 247 weapons , 250 @,@ 000 musket cartridges , and 520 @,@ 000 percussion caps , as well as the four bronze cannon of Totten 's battery . Long arms in the Arsenal 's inventory consisted of :`
- noisy: `Inse th Little Rock Aenal after it seiure inFebrua, 1861 , the Confderates inventoie oe 10 @,@ 24 epons , 250 @,@ 000 msket cartridges  and520 @,@ 000 percussion caps  aswellas he for bronze cannon of Ttten 'sbatery . Log ars n he Arenal 's inventory consiste of :`
- MED 36 (normalized 0.120); tokens 69 -> 84 (change +15)
- 36 edit(s), token count +15, continuation pieces +22.

### wt2-train-000075  (substitution, intensity 0.15)
- clean: `M1822 .69 cal ( percussion @-@ converted ) 53`
- noisy: `M1z22 .6cpcblu( peicujsiok @-r converted ) 53`
- MED 9 (normalized 0.200); tokens 14 -> 20 (change +6)
- 9 edit(s), token count +6, continuation pieces +7.

### wt2-train-000076  (insertion, intensity 0.05)
- clean: `M1842 .69 cal smoothbore ( percussion ) 357`
- noisy: `M1842 .69 cal smoothbore ( percussion ) 357`
- MED 0 (normalized 0.000); tokens 13 -> 13 (change +0)
- intensity 0.05: no character positions were sampled for this short text, tokenization identical.
