all:
	pip install .
	python3 tests/scripts/test_dataset.py -d CICIDS_2017
	python3 tests/scripts/test_dataset.py -d CIDDS-001  
	python3 tests/scripts/test_dataset.py -d Kitsune_Active_Wiretap  
	python3 tests/scripts/test_dataset.py -d Kitsune_Fuzzing  
	python3 tests/scripts/test_dataset.py -d Kitsune_Mirai_hyp 
	python3 tests/scripts/test_dataset.py -d Kitsune_SSDP_Flood         
	python3 tests/scripts/test_dataset.py -d Kitsune_SYN_DoS
	python3 tests/scripts/test_dataset.py -d Kitsune_ARP_MitM        
	python3 tests/scripts/test_dataset.py -d Kitsune_Mirai    
	python3 tests/scripts/test_dataset.py -d Kitsune_OS_Scan        
	python3 tests/scripts/test_dataset.py -d Kitsune_SSL_Renegotiation  
	python3 tests/scripts/test_dataset.py -d Kitsune_Video_Injection