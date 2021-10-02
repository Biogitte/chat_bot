First Aid Chatbot
-----------------

The first aid Chatbot are based on a Sequential Deep Learning model trained on [a first aid intents Kaggle dataset from therealsampat](https://www.kaggle.com/therealsampat/intents-for-first-aid-recommendations).

Repository structure
--------------------
    .
    ├── README.md       # README file
    ├── data        # Data directory
    │	├── output      # Output files
    │	│	├── <date>_chatbot.h5       # Chatbot model
    │	│	├── <date>_chatbot_classes.pkl      # Picled list of Chatbot categories/classes
    │	│	├── <date>_chatbot_words.pkl        # Pickled Chatbot vocabulary list
    │	│	├── <date>_learning_curves.png      # Learning curves from Chatbot model training
    │	│	└── <date>_history_logger.csv       # Training history (i.e., epochs, )
    │	└── training        # Input training data
    │	     └── first_aid_intents.json     # Input first aid data set from Kaggle
    ├── exec        # Executables
    │	├── fetch_data.sh       # Fetch Kaggle data: $FETCH_DATA
    │	├── run_bot.sh      # Run the chatbot: $CHATBOT
    │	└── train_bot.sh        # Train Chatbot model: $TRAIN_BOT
    ├── global_env.sh       # Global environment variables
    ├── requirements.txt        # Dependencies
    ├── setup.py        # Setup script to install local modules
    └──  src        # Source code
	    ├── __init__.py     # __init__ file for module creation
	    ├── chatbot.py      # Chatbot script
	    ├── fetch_kaggle_data.py        # Fetch Kaggle data
	    └── train_bot.py        # Model training script

Prerequisites
-------------
* Python 3.x
* Set up Kaggle authentication: on your Kaggle account, under API, select `Create New API Token`, and a `kaggle.json` file will be downloaded on your computer. Move this file to `~/.kaggle/kaggle.json` on MacOS/Linux. Remember to run `chmod 600 ~/.kaggle/kaggle.json` to ensure it is only readable for you.
* Mac OSX users: Run `brew install libomp` to install OpenMP runtime (for Xboost). This step requires that homebrew is installed.
* Adjust the global environment variables in "global_env.sh" accordingly to your use case.

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

Ensure that the input JSON file being used follows this structure (or change it to the new format in the /src/train_bot.py script):

    { "intents": 
    [{  
        "tag": "<tag/category>",
        "patterns": ["Input question", "Input question", "Input question"],
        "repsonses": ["Response", "Response"]  
        },
    {...}     
    ]}

**Outputs include:**
* The resulting Chatbot model: `/data/output/<date>_chatbot.h5`
* Learning curves: `/data/output/<date>_learning_curves.png`
* History data for model training (i.e., epochs, accuracies and losses for train and test data): `/data/output/<date>_history_logger.csv`
* (i.e., `Words`): `/data/output/<date>_chatbot_words.pkl`
* List of Chatbot categories/classes (i.e., `Classes`): `/data/output/<date>_chatbot_classes.pkl`

Run the Chatbot
---------------
    source global_env.sh
    sh $CHATBOT
