git clone https://github.com/Zainab-Saad/fl-energy 
cd fl-energy 

# make the virtual environment 
python3 -m venv venv  

# checkout into the virtual env 
source venv/bin/activate  

# install the project requirements 
pip install -r requirements.txt   

# test if code is setup before starting the experiments, since experiments are run on gpu so set gpu flag 
# try the baseline model training 
python src/baseline_main.py --model=mlp --dataset=mnist --gpu=0 --epochs=10 

# try federated model training 
python src/federated_main.py --model=mlp --dataset=mnist --gpu=0 --iid=1 --epochs=10 

# run the baseline code for question 1
python src/baseline_main.py --model=mlp --dataset=mnist --optimizer=sgd --lr=0.01 --seed=1 --gpu=0 


# run the shell script for question 1 --- this would run the federated_main.py with different arguments to study impact of changing fraction of clients C 
cd src  
bash run_q1_mlp.sh 

# run the shell script for question 2 --- this would run the federated_main.py with different arguments to study impact of changing local epochs E and batch size B 
cd src 
bash run_q2_mlp.sh 

# make the plots for question 1 and 2 by running the cells in files plot_q1.ipynb and plot_q2.ipynb 

# --------------------------------------------------------------------------------------------------
# format the mlp architecture used
MLP(
  (layer_input): Linear(in_features=784, out_features=64, bias=True)
  (relu): ReLU()
  (dropout): Dropout(p=0.5, inplace=False)
  (layer_hidden): Linear(in_features=64, out_features=10, bias=True)
  (softmax): Softmax(dim=1)
)