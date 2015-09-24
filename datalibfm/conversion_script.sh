ipython -m source_py.libfm_input datalibfm/raw_output.txt
sed 's/\"/ /g' datalibfm/raw_output.txt > datalibfm/output.txt 
