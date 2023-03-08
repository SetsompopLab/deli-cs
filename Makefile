.PHONY: conda update pip data data+ clean

conda:
	conda env create -f environment.yaml

update:
	conda env update --file environment.yaml --prune

pip:
	pip install git+https://github.com/mikgroup/sigpy.git@master
	pip install pytorch-lightning
	pip install git+https://gitlab.com/dvolgyes/zenodo_get

data:
	zenodo_get -d 10.5281/zenodo.7703200  # test data + dependencies only
	mkdir -p data/testing/
	mkdir -p data/shared/
	for case in 000 001 002 003 004; do
		tar -xzvf teast_case${case}.tar.gz
	done
	tar -xzvf shared.tar.gz -C data/

data+:
	zenodo_get -d 10.5281/zenodo.7703200 # test data + dependencies only
	mkdir -p data/testing/
	mkdir -p data/shared/
	for case in 000 001 002 003 004; do
		tar -xzvf test_case${case}.tar.gz
	done
	tar -xzvf shared.tar.gz -C data/

	zenodo_get -d 10.5281/zenodo.7697373 # training and validation data
	mkdir -p data/training/
	mkdir -p data/validation/
	for case in 000 001 002 003 004 005 006 007 008 009; do
		tar -xzvf train_case${case}.tar.gz
	done

	for case in 000 001; do
		tar -xzvf val_case${case}.tar.gz
	done
		
docker:
	if [ ! -d "MRF" ] ; then  git clone https://github.com/SetsompopLab/MRF.git ; fi

	$(MAKE) -C MRF/src/01_calib
	$(MAKE) -C MRF/src/02_recon

clean:
	rm -rf __pycache__
	rm -rf .ipynb_checkpoints
	rm -rf MRF
	conda env remove -n deliCS
