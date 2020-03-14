# rumor
1. rumor3.12.csv and rumor_keywords3.12.csv are the data I collect the data in 2020.3.14.

2. false.csv includes all the rumors that are identified as flase. true.csv, suspect.csv are the same.

3. top_ten_hot_words.txt give top ten hot words in each case

4. main.py is the code for the project. Some major functions are:
  I. SpiderRumor  which crawl the data from web site
  II. SpiderTrans  which can translate chinese into english
  III.label, input_txt= read_data(filename)
  IV.one_hot_class which use binary classifier to train the data
  V. lstm which train a neural network

5. rumor_chinese.csv and rumor_chinese_keword.csv are the orignal data I collected using UTF-8 coding

6. other files includes some data trained by lstm work.
