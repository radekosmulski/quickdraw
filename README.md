# Quick, Draw! Kaggle Competition Starter Pack v2

The code in this repo is all you need to make a first submission to the [Quick, Draw! Kaggle Competition](https://www.kaggle.com/c/quickdraw-doodle-recognition). It uses the [FastAi library](https://github.com/fastai/fastai).

For additional information please refer to the discussion thread on [Kaggle forums](https://www.kaggle.com/c/quickdraw-doodle-recognition/discussion/69409).

I provide instructions on how to run the code below.

**This code is based on code from a fast.ai MOOC that will be publicly available in Jan 2019**

*You can find an earlier version of this starter pack [here](https://github.com/radekosmulski/quickdraw/tree/v1). This iteration eliminates some of the rough edges that resulted in a relatively low score and also introduces the data block API. A major change is that here I am generating drawings on the fly - this should help with experimentation.*

This version of the code runs with fastai 1.0.27 and is likely incompatible with more recent releases.

## Making the first submission
1. You need to have the FastAi library up and running. You can find installation instructions [here](https://github.com/fastai/fastai#installation).
2. You will also need to download the competition data. The competition can be found under this [url](https://www.kaggle.com/c/quickdraw-doodle-recognition). If you do not have a Kaggle account you will need to register. There are many ways to download the data - I use the [Kaggle CLI](https://github.com/Kaggle/kaggle-api). To set it up you will need to generate an API key - all this information is available in the [Kaggle CLI repository](https://github.com/Kaggle/kaggle-api). If you do not want to set this up at this point, you can download the data from the [competition's data tab](https://www.kaggle.com/c/quickdraw-doodle-recognition/data).
3. We will need `test_simplified.csv` and `train_simplified.zip`. Please put them in the `data` directory. If you were to download one of the files using the Kaggle CLI, the command (executed from the root of the repository) would be `kaggle competitions download -c quickdraw-doodle-recognition -f test_simplified.csv -p data`.
4. We now need to unzip the downloaded files. cd into the data folder and create a new directory `train`. Extract the downloaded files by executing `unzip train_simplified.zip -d train`.
6. Open `first_submission.ipynb`. If you have kaggle CLI installed and configured, you can uncomment the last line. Hit run all. See you on the leaderboards :)
