# Test_task-semantic-segmentation-
I started solving the problem of semantic segmentation by decoding RLE lists into binary masks and generating a feature file with them. Converting the RLE format into masks and creating a complete file for training the model, presented in the notebook RLE_endcoder.

The file Unet.py presents the complete stage of loading, data normalization, as well as the formation of the architecture Unet with subsequent model training. At the end, the model is saved in the Model_list folder.

