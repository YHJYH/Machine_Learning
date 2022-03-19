This file is to provide the instructions to run codes in order to reproduce all works for the course work. 

Note that dataset we used is "data_new" in file folder dataset-oxpet, which should be downloaded from [here](https://weisslab.cs.ucl.ac.uk/WEISSTeaching/datasets/-/tree/oxpet/). In our submission, we don't include this file as it is too large to submit. Anyone who needs the dataset should download themselves.

After zip file dataset-oxpet is downloaded, unzip it and place it in "cw2" file folder

After unzipping the .zip file and downloading dataset-oxpet, the structure of folder 'cw2' should be shown as below:<br>
-------------------------------------------<br>
cw2<br>
 |--- instructions.txt<br>
 |--- requirements.txt<br>
 |--- dataset-oxpet<br>
 |--- single<br>
 |      |--- bounding_box.py<br>
 |      |--- classification.py<br>
 |      |--- segmentation.py<br>
 |      |--- data_loader.py<br>
 |--- mmoe<br>
 |     |--- mmoe.py<br>
 |     |--- mmoe_oeq.py<br>
 |     |--- data_loader.py	<br>
 |--- ablation<br>
 |       |--- no_cls.py<br>
 |       |--- no_bbox.py<br>
 |       |--- data_loader.py<br>
--------------------------------------------<br>

The first step is to run requirements file to properly install all packages used in tasks.<br>
Then run 'bounding_box.py', 'classification.py', 'segmentation.py' files respectively to get baseline results.<br>
Next, run 'mmoe.py' to get the basic result of MTL training score; run 'mmoe_oeq.py' to get the result of OEQ.<br>
Finally, run 'no_bbox.py' and 'no_cls.py' to get results of two ablation results.<br>
