ipython -m source_py.libfm_input datalibfm/raw_train_output.txt datalibfm/raw_test_output.txt
sed 's/\"/ /g' datalibfm/raw_train_output.txt > datalibfm/train_output.txt 
sed 's/\"/ /g' datalibfm/raw_test_output.txt > datalibfm/test_output.txt 
