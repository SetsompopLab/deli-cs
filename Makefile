.PHONY: conda update pip data data+ clean

conda:
	conda env create -f environment.yaml

update:
	conda env update --file environment.yaml --prune

data:
	zenodo_get -d 10.5281/zenodo.7734431
	for case in 000 001 002 003 004; do
		tar -xzvf case${case}_preprocessed.tar.gz
	done
	tar -xzvf bartcompare.tar.gz

data+:
	$(MAKE) data
	zenodo_get -d 10.5281/zenodo.7703200 
	for case in 000 001 002 003 004; do
		tar -xzvf test_case${case}.tar.gz
	done
	tar -xzvf train_case000.tar.gz
	tar -xzvf shared.tar.gz
	tar -xzvf checkpoints.tar.gz

data++:
	$(MAKE) data
	$(MAKE) data+

	zenodo_get -d 10.5281/zenodo.7697373
	for case in 000 001 002 003 004 005 006 007 008 009; do
		tar -xzvf train_case${case}.tar.gz
	done

	for case in 000 001; do
		tar -xzvf val_case${case}.tar.gz
	done
		
docker:
	if [ ! -d "MRF" ] ; then  git clone https://github.com/SetsompopLab/MRF.git ; fi
	git -C MRF checkout d480dbc 
	$(MAKE) -C MRF/src/01_calib
	$(MAKE) -C MRF/src/02_recon

clean:
	rm -rf __pycache__
	rm -rf .ipynb_checkpoints
	rm -rf MRF
	conda env remove -n deliCS
