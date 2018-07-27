

Windows example

python afs.py -d C:\data\kaggle\credit_card_fraud\creditcard.csv -c afs_kaggle_cc.yaml -o C:\data\kaggle\credit_card_fraud\populations\201802 -r afs_run_000_epoch

Linux example

python afs.py -d /datastorage02/datasets/kaggle/credit_card_fraud/creditcard.csv -c afs_kaggle_cc_demo.yaml -o /datastorage02/datasets/kaggle/credit_card_fraud/populations -r afs_run_000_epoch


(not so) Authoritative list of tuning options:

Randomization
- Random seed
- Tree-based algorithm seed
- CV split seed

GP algorithm choice
- choice of evolutionary algorithm
- choice of operators
-

GP Population params
- population size
- offspring size
- mate, mutate, select methods
- population initialization method
- fitness evaluators
- whether to minimize or maximize fitness function
- which stats to record
- hof size

Best Feature Set params
- MIN_BFS_SIZE = 5            # Don't allow the BFS to shrink beyond this size
- MAX_BFS_SIZE = 50           # Don't allow the BFS to grow beyond this size
- MAX_BFS_REMOVE_EVALS = 40   # Number of features to evaluate from the remove queue
- MAX_BFS_ADD_EVALS = 40      # Number of features to evaluate from the add queue
- ADD_THRESHOLD = 0.0005      # The higher the threshold, the more difficult it is to add features
- REMOVE_THRESHOLD = 0.001    # The higher the threshold, the more easily features get dropped
- REPLACE_THRESHOLD = 0.0     # The higher the threshold, the more difficult it is
- N_FOLDS = 5                 # Number of CV folds for BFS wrapper scoring method
- choice of scoring algorithm