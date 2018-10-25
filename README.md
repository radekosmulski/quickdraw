# Quick, Draw! Kaggle Competition Starter Pack

The code in this repo is all you need to make a first submission to the [Quick, Draw! Kaggle Competition](https://www.kaggle.com/c/quickdraw-doodle-recognition). It uses the [FastAi library](https://github.com/fastai/fastai).

I provide instructions on how to run the code below.

**This code is based on code from a fast.ai MOOC that will be publicly available in Jan 2019**

## Making the first submission
1. You need to have the FastAi library up and running. You can find installation instructions [here](https://github.com/fastai/fastai#installation).
2. You will also need to download the competition data. The competition can be found under this [url](https://www.kaggle.com/c/quickdraw-doodle-recognition). If you do not have a Kaggle account you will need to register. There are many ways to download the data - I use the [Kaggle CLI](https://github.com/Kaggle/kaggle-api). To set it up you will need to generate an API key - all this information is available in the [Kaggle CLI repository](https://github.com/Kaggle/kaggle-api). If you do not want to set this up at this point, you can download the data from the [competition's data tab](https://www.kaggle.com/c/quickdraw-doodle-recognition/data).
3. We will need `test_simplified.csv` and `train_simplified.zip`. Please put them in the `data` directory. If you were to download one of the files using the Kaggle CLI, the command (executed from the root of the repository) would be `kaggle competitions download -c quickdraw-doodle-recognition -f test_simplified.csv -p data`.
4. We now need to unzip the downloaded files. cd into the data folder and create a new directory `train`. Extract the downloaded files by executing `unzip train_simplified.zip -d train`.
5. Let's prepare our data now. Start the `jupyter notebook` server and open `data_preparation.ipynb` and run all.
6. Now that we have the data, let's train. Open `first_submission.ipynb`. If you have kaggle CLI installed and configured, you can uncomment the last line and hit run all. See you on the leaderboards :)
