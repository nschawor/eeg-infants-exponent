preprocessing:
	  python3 proc_00_reformat_eeg_data.py
		python3 proc_01_compute_age_for_each_session.py
		python3 proc_02_clean_eeg_data_ica.py
		python3 proc_03_compute_spatial_patterns.py

exponent_analysis:
		python3 proc_10_exponent_computation.py
		python3 proc_11_exponent_combine_results.py
		Rscript proc_12_exponent_linear_model.R
		python3 fig1_exponent_example_1subject.py
		python3 fig2_exponent_summary.py

burst_analysis:
	  python3 proc_20_bursts_compute_cycle_features.py
		python3 proc_21_bursts_combine_results.py
		Rscript proc_22_bursts_model.R
		Rscript proc_23_bursts_model_compare_motor_vs_parietal.R
		python3 fig3_bursts_example_1subject.py
		python3 fig4_bursts_summary.py


all: preprocessing exponent_analysis burst_analysis
