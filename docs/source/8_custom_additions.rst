#####################
Custom additions
#####################

MASTML contains some avenues for customization.

*******************
Custom data setup
*******************

The input file can be set up to run custom code that creates main CSV files for use in the rest of the tests.

This functionality may be helpful if:

* Several CSV files from different sources change and need to be routinely collated
* Large amounts of custom pre-processing are needed on a changing source dataset
* Changes to an original dataset must be tracked

The user must create this custom code as a class with a run method.
This class can be put in the ``custom_data`` folder, and a corresponding ``CSV Setup`` section may be created in the input file.

Example::

    [CSV Setup]
        setup_class = custom_data.DBTTDataCSVSetup
        save_path = ../exports
        import_path = ../imports_201704

* **setup_class**: The format for this keyword is ``custom_data.<package name (.py file)>``, where the package name must be the same as the class name
    * The module file must contain a single class with the same name as the module name.
    * The class must have a ``run`` method which sets up a CSV file.

* All other keywords should correspond to parameters in the ``__init__`` method of the custom class.

*****************
Custom features
*****************

Custom features may be needed for generation on the fly, especially if their generation involves tunable hyperparameters.

Custom feature code may be put in the ``custom_features`` folder. See ``custom_features.Testing`` for an example.

* Each custom feature class should correspond to a distinct data topic. 
* Within each custom feature class, the __init__ method should take a dataframe and define self.df (copy the __init__ method from custom_features.Testing)
* Each class may have multiple methods, corresponding to different features.
* Each separate feature method should take in named arguments, which will be accessed through the input file (see for example :doc:`4_p1_paramgridsearch`), and also take in ``**params``.
* Each feature should return a pandas Series. 

********************
Custom models
********************

Custom models may be created in the ``custom_models`` folder, based off of ``custom_models.BaseCustomModel`` and taking ``**kwargs`` in the __init__ method.

Custom models may be accessed through the :ref:`Model Parameters <model-parameters>` section of the input file.

Only one custom model may be used per input file.

Example:: 

    [[Model Parameters]]

        [[custom_model]]
        package_name = custom_models.DBTT_E900
        class_name = E900model
        wtP_feature = wt_percent_P
        wtNi_feature = wt_percent_Ni
        wtMn_feature = wt_percent_Mn
        wtCu_feature = wt_percent_Cu
        fluence_n_cm2_feature = fluence_n_cm2
        temp_C_feature = temperature_C
        product_id_feature = product_id

* **package_name**: Package name; format is ``custom_models.<package name (.py file)>``
* **class_name**: Class name of the custom model. Each package can have several custom models, each defined as a separate class.
* All other keywords should be defined in the __init__ method of the model.
