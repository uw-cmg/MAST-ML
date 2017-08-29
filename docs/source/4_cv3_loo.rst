=====================================
LeaveOneOutCV
=====================================

Perform leave-one-out (LOO) cross validation.

LOO cross validation can be considered a special case of :doc:`4_cv2_kfold`, where the number of folds N is equal to the number of data points, and there is only a single N-fold test containing N subtests.

-----------------
Input keywords
-----------------

See :doc:`4_cv2_kfold`: 

* training_dataset (Should be the same as testing_dataset)
* testing_dataset (Should be the same as training_dataset)
* xlabel 
* ylabel

Additional keywords:

* mark_outlying_points: Number of outlying points to mark. Use only one number.

----------------
Code
----------------

.. autoclass:: LeaveOneOutCV.LeaveOneOutCV

