Prerequisites
-------------
* Python 3.x
* Set up Kaggle authentication: on your Kaggle account, under API, select `Create New API Token`, and a `kaggle.json` file will be downloaded on your computer. Move this file to `~/.kaggle/kaggle.json` on MacOS/Linux. Remember to run `chmod 600 ~/.kaggle/kaggle.json` to ensure it is only readable for you.
* Mac OSX users: Run `brew install libomp` to install OpenMP runtime (for Xboost). This step requires that homebrew has been installed.

Get started
-----------
     # install virtualenv
     pip3 install virtualenv
     
     # create a virtual environment
     virtualenv venv --python=<path-to-python-3.*>
     
     # activate environment
     source venv/bin/activate
     
     # install requirements
     pip3 install -r requirements.txt
     
     # set the global environment variables
     source global_env.sh
     
     # install local python packages
     python3 setup.py install
     pip3 install -e .

Download data and train Chatbot model
-------------------------------------

Kaggle data set used in the example: [the first aid intents dataset from therealsampat](https://www.kaggle.com/therealsampat/intents-for-first-aid-recommendations)

If you want to train a Chatbot model using another dataset from Kaggle, do the following.

    # 1) Edit global_env.sh in the top-level directory accordingly to your new dataset
    # 2) Reset the global environment variables (i.e., $KAGGLE_DATASET, $KAGGLE_NEW_NAME, and $INTENTS)
    source global_env.sh
    # 3) Download the data you specified in the global_env
    sh $FETCH_DATA
    # 4) Train the model
    sh $TRAIN_BOT

Ensure that the input JSON file being used follows this structure (or, alternatively, change it to the new format in the /src/train_bot.py script):

    { "intents": [
        {  
        "tag": "<tag/category>",
        "patterns": ["Input question", "Input question", "Input question"],
        "repsonses": ["Response", "Response"]  
        },
        
        {  
        "tag": "<tag/category>",
        "patterns": ["Input question", "Input question", "Input question"],
        "repsonses": ["Response", "Response"]  
        }, 
    ]  
    }


Run the Chatbot
---------------
    source global_env.sh
    sh $CHATBOT
