# LSTM RNN for Next Word Prediction + Streamlit Application : Harry Potter Edition 🪄

## Dataset and Data Filtering:

- The dataset is available to download on [Kaggle](https://www.kaggle.com/datasets/gulsahdemiryurek/harry-potter-dataset). It contains 7 Comma Separated Files out of which only 3 have been used for training and testing of the model.
- The data used is essentially lines from the Harry Potter movies (Part 1-3). Each row is a line from the script by a particular character.
- The data was specifically filtered to include lines exclusively from the three main characters of the Harry Potter franchise—Harry, Ron, and Hermione. This selection was driven by both the characters' popularity and the dataset's significantly higher volume of entries for these individuals
- The three CSV files were combined horizontally using  the ```pd.concat()``` method and then ``.isin()`` method from Pandas.

## Preprocessing:
- The ``Tokenizer`` object from the text preprocessing module in Keras was used for most of the data preprocessing and preparation part.
- The input sentences were converted into sequences of integer indexes by using text to sequences methods available in the Keras preprocessing module for text preprocessing
- To add semantic information, the n-grams for each sentence were added to the final corpus of input sequences after they were padded to make their lengths equal.
- Finally, the data was transformed into input features X  and output features y by slicing each sequence vector till its last element. 
- The final train test split was done where the ratio was 80-20.
- The output feature array y, which was a 2D array containing the label/word index for the sequence in a 1D array was transformed into a 2D array of shape ``num of inputs x total_words`` to convert the output feature array y from a continuous variable into a categorical variable using the utility method ``to_categorical()``.

## Model Architecture & Training:

- The model is essentially a 2 layer LSTM network.
- The complete architecture contains:
  - An Embedding Layer with 100 output dimensions
  - An LSTM layer with 150 units
  - A Dropout layer with 20% dropout rate to minimize the chances of overfitting.
  - An LSTM layer with 100 units
  - A Dense layer with softmax activation.
    
    ![architecture](https://github.com/user-attachments/assets/2380fd4b-6993-458e-b9d5-f943c587cf83)

 
- The model was compiled to optimize  the ``categorical_crossentropy`` loss using the ``Adam`` optimizer.
- Training was done on GPU for 100 epochs. Training duration was ~ 8 minutes.

## Testing for new data on the Streamlit Application:

- 

