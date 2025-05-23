"""
    Interface to calculate the MALD Importance and W Statistics given a `PredictionModel`, either your own or created through ``maldImportance.superBasicNetworks``; see :doc:`superBasicNetworks`.
    
    If you do wish to use a ``maldImportance.superBasicNetworks.SimpleNN`` then you can use the ``maldImportance.nnImportance`` (:doc:`nnImportance`) module for simplicity.
"""

import pandas as pd
import numpy as np

from typing import Callable, Literal, Protocol, Self
from abc import abstractmethod

class PredictionModel( Protocol ):
    """
        Interface for prediction models, including ``hexathello.superBasicNetworks.SimpleNN``, as well as most tensorflow keras network predictors.
    """
    @abstractmethod
    def fit( self: Self, X, y, **kwargs ) -> Self:
        """
            :param X: Explanatory data, likely concatenated with knockoffs
            :param y: Outcome data
            :param kwargs: Other arguments
        """
        raise NotImplementedError()
    #
    
    @abstractmethod
    def predict( self: Self, X ) -> np.ndarray:
        """
            :param X: Explanatory data, likely concatenated with knockoffs
            :returns: Predictions, one for each row of `X`
            :rtype: np.ndarray
        """
        raise NotImplementedError()
    #
    
    @abstractmethod
    def call( self: Self, X ):
        """
            :param X: Explanatory data, likely concatenated with knockoffs
            :returns: A prediction result
            
            Used for auto differentiation
        """
        raise NotImplementedError()
    #
#/class PredictionModel( Protocol )

def auto_diff(
    model: PredictionModel,
    X: np.ndarray
    ) -> np.ndarray:
    """
        :param PredictionModel model: Predictor which can use autodifferentiation
        :param np.ndarray X: Explanatory data, likely including the knockoffs
        :returns: Array of partial derivatives for each variable at each data point. Same dimensions as `X`
        :rtype: np.ndarray
        
        Uses the auto differentiating capabilities of our `model` to get the exact MALD values.
    """
    import tensorflow as tf
    
    _X = tf.constant( X )
    
    y_pred = model.predict( _X )
    
    tape: tf.GradientTape
    with tf.GradientTape() as tape:
        tape.watch( _X )
        y_hat: tf.Tensor = model.call( _X )
    #
    
    return tape.gradient( y_hat, _X ).numpy()
#/def auto_diff

def _localGrad_forNumeric(
    j: int,
    X: np.ndarray,
    y_hat: np.ndarray,
    model: PredictionModel,
    bandwidth: float
    ) -> np.ndarray:
    """
        Get the bandwidth local gradient approximation for variable j
        X: all data
        y_hat: The base prediction, result of `model.predict(X)`
        model: PredictionModel already fit and trained
        bandwidth: Exact literal value
    """
    # Set the X + bandwidth matrix
    X_epsilon: np.ndarray = np.copy( X )
    X_epsilon[:, j ] += bandwidth
    
    # Get the approximation of local gradient via the definition of the derivative
    return ( model.predict(X_epsilon) - y_hat )/bandwidth
#/def _localGrad_forNumeric

def _localGrad_forCategories(
    j: list[ int ],
    X: np.ndarray,
    model: PredictionModel,
    drop_first: bool
    ) -> np.ndarray:
    """
        Get the change in prediction for a group of category columns by changing each of the values to. 1, and the others to 0
        
        If drop_first, it gets compared with setting all to 0.
    """
    _X: np.ndarray = np.copy( X )
    
    y_out_dim: int = len( j )
    if drop_first: y_out_dim += 1
    
    y_out: np.ndarray = np.zeros( shape = (X.shape[0], y_out_dim ) )
    
    # Get the predicted values at each test category
    for h in range( len(j) ):
        # Reset all to 0, set one to 1
        _X[ :, j ] = 0
        _X[ :, j[h] ] = 1
        
        y_out[ :, h ] = model.predict( _X )
    #
    
    # Use last index setting all to 0
    if drop_first:
        _X[ :, j ] = 0
        y_out[ :, -1 ] = model.predict( _X )
    #
    
    return np.max( y_out, axis = 1 ) - np.min( y_out, axis = 1 )
#/def _localGrad_forCategories

def importancesFromModel(
    model: PredictionModel,
    X: np.ndarray | pd.DataFrame,
    Xk: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    local_grad_method: Literal['auto_diff','bandwidth'] = 'auto_diff',
    fit: bool = True,
    bandwidth: float | None = None,
    exponent: float = 2.0,
    drop_first: bool = True,
    verbose: int = 0
    ) -> np.ndarray:
    """
        :param PredictionModel model: Predictor which can use autodifferentiation
        :param np.ndarray|pd.DataFrame X: Explanatory data
        :param np.ndarray|pd.DataFrame Xk: Knockoff explanatory data
        :param np.ndarray|pd.Series y: Outcome data
        :param Literal['auto_diff','bandwidth'] local_grad_method: Method of MALD. Defaults to `'auto_diff'` for exact autodifferentiation. `'bandwidth'` uses the bandwidth approximation when auto differentiation is not available.
        :param bool fit: Whether to fit `model`. Set to `False` if you have already trained it to your satisfaction on the combined explanatory and knockoff data together.
        :param float|None bandwidth: Width if `local_grad_method = 'bandwidth'`
        :param float exponent: Power to take of each MALD value. 1.0 and 2.0 both work reasonably well.
        :param bool drop_first: How to handle one-hot-encoding categorical variables. If `True` the number of associated columns is the number of categories minus 1.
        :param int verbose: How much to print out, for mostly for debugging.
        :returns: Array of importances, with length equal to twice the width of `X`
        :rtype: np.ndarray
        
        Takes an initialized PredictionModel, likely fits it, and gets the MALD importances for each `X` and `Xk` variable
        
    """
    assert X.shape == Xk.shape
    
    # Check for categorical variables, one hot encode variables if necessary
    
    oheDict_X: dict[ int, int | list[ int ] ]
    oheDict_Xk: dict[ int, int | list[ int ] ]
    _X: np.ndarray
    _Xk: np.ndarray
    
    if isinstance( X, pd.DataFrame ):
        assert all( X.dtypes.iloc[j] == Xk.dtypes.iloc[j] for j in range( X.shape[1] ) )
        from . import utilities
        
        if any( dtype == 'category' for dtype in X.dtypes ):
            _X = pd.get_dummies(
                X,
                drop_first = drop_first
            ).to_numpy( dtype = float )
            
            _Xk = pd.get_dummies(
                Xk,
                drop_first = drop_first
            ).to_numpy( dtype = float )
            
            oheDict_X = utilities.get_oheDict(
                X = X,
                drop_first = drop_first,
                starting_index = 0
            )
            oheDict_Xk = utilities.get_oheDict(
                X = Xk,
                drop_first = drop_first,
                starting_index = X.shape[1],
                starting_ohe_index = _X.shape[1]
            )
        #
        else:
            oheDict_X = {}
            oheDict_Xk = {}
            _X = X.to_numpy()
            _Xk = Xk.to_numpy()
        #/if any( dtype == 'category' for dtype in X.dtypes )
    #/if isinstance( X, pd.DataFrame )
    else:
        oheDict_X = {}
        oheDict_Xk = {}
        _X = X
        _Xk = Xk
    #/if isinstance( X, pd.DataFrame )/else
    
    oheDict: dict[ int, int | list[int] ] = oheDict_X | oheDict_Xk
    
    X_concat: np.ndarray = np.concatenate(
        [_X, _Xk ], axis = 1
    )
    
    _y: np.ndarray
    if isinstance( y, pd.Series | pd.DataFrame ):
        _y = y.to_numpy()
    #
    else:
        _y = y
    #/if isinstance( y, pd.Series )/else
    _y = np.reshape( _y, ( X_concat.shape[0], ) )
    
    # Fit if necessary; we can have a model already trained
    #   by setting to False
    if fit:
        model.fit( X_concat, _y, )
    #/if fit
    
    
    auto_diff_matrix: np.ndarray | None
    y_hat: np.ndarray | None
    if local_grad_method == 'auto_diff':
        auto_diff_matrix: np.ndarray = auto_diff(
            model,
            X_concat
        )
        y_hat = None
    #
    elif local_grad_method == 'bandwidth':
        auto_diff_matrix = None
        y_hat = model.predict( X_concat )
        
        if False:
            # TEST 2025-02-15
            print( _y )
            print( y_hat )
            
            _y_diff = y_hat - _y
            print( "_y_diff: {} ({})".format(np.mean(_y_diff),np.std(_y_diff)))
            raise Exception("_y_diff")
        #
        
        if bandwidth is None:
            bandwidth = X_concat.shape[0]**(-0.2)
        #/if bandwidth is None
    #
    else:
        raise Exception("Unrecognized local_grad_method={}".format(local_grad_method))
    #/switch local_grad_method

    p_out: int = X.shape[1] + Xk.shape[1]
    localGrad_matrix: np.ndarray

    if oheDict == {}:
        # all numeric
        if local_grad_method == 'auto_diff':
            localGrad_matrix = auto_diff_matrix
        #
        elif local_grad_method == 'bandwidth':
            localGrad_matrix = np.zeros(
                shape = X_concat.shape
            )
            for j in range( X_concat.shape[1] ):
                localGrad_matrix[ :, j ] = _localGrad_forNumeric(
                    j = j,
                    X = X_concat,
                    y_hat = y_hat,
                    model = model,
                    bandwidth = bandwidth
                )
            #
        #
        else:
            raise Exception("Unrecognized local_grad_method={}".format(local_grad_method))
        #/switch local_grad_method
    #/if oheDict == {}
    else:
        # Some categories
        localGrad_matrix: np.ndarray = np.zeros(
            shape = ( X.shape[0], p_out )
        )
        for j in range( p_out ):
            if isinstance( oheDict[j], int ):
                # numeric
                if local_grad_method == 'auto_diff':
                    column_grad = auto_diff_matrix[:, oheDict[j] ]
                    #print("grad {}".format(j))
                    #print( column_grad )
                    localGrad_matrix[:,j] = column_grad
                #
                elif local_grad_method == 'bandwidth':
                    column_grad = _localGrad_forNumeric(
                        j = oheDict[j],
                        X = X_concat,
                        y_hat = y_hat,
                        model = model,
                        bandwidth = bandwidth
                    )
                    
                    localGrad_matrix[ :, j ] = column_grad
                #
                else:
                    raise Exception("Unrecognized local_grad_method={}".format(local_grad_method))
                #/switch local_grad_method
            #
            else:
                # category
                column_grad = _localGrad_forCategories(
                    j = oheDict[j],
                    X = X_concat,
                    model = model,
                    drop_first = drop_first
                )
                localGrad_matrix[:,j] = column_grad
            #/if isinstance( oheDict[j], int )/else
            #print( localGrad_matrix[ :, j ] )
        #/for j in range( p_out )
    #/if oheDict == {}/else
    
    importances: np.ndarray = np.mean(
        np.abs( localGrad_matrix )**exponent,
        axis = 0
    )
    
    # Fix the shit: throw an error if it's a zero matrix
    if np.allclose(
        importances, 0
    ):
        raise Exception("Got zeros importances")
    #
    
    return importances
#/def importancesFromModel

def wFromImportances(
    importances: np.ndarray,
    W_method: Literal['difference','signed_max'] = 'difference',
    verbose: int = 0
    ) -> np.ndarray:
    """
        :param np.ndarray importances: Importance measures, likely from ``importancesFromModel()``
        :param Literal['difference','signed_max'] W_method: How to calculate W statistics from importance measures, given the two most common methods.
        :param int verbose: How much to print out, for mostly for debugging.
        :returns: W statistics, half the length of `importances`, the same length as the original number of variables
        :rtype: np.ndarray
        
        Converts arbitrary importances to W statistics for the knockoff procedure.
    """
    p: int = len( importances ) // 2
    W_out: np.ndarray
    if W_method == 'difference':
        W_out = importances[ : p ] - importances[ p: ]
    #
    elif W_method == 'signed_max':
        W_out = np.zeros( shape = ( p,) )
        for j in range(p):
            if importances[ j ] > importances[ j+p ]:
                W_out[ j ] = importances[ j ]
            #
            elif importances[ j ] < importances[ j+p ]:
                W_out[ j ] = importances[ j+p ]
            #/switch importances[ j ] - importances[ j+p ]
        #/for j in range(p)
    else:
        raise ValueError("Unrecognized W_method={}".format(W_method))
    #
    return W_out
#/def wFromImportances

def wFromModel(
    model: PredictionModel,
    X: np.ndarray | pd.DataFrame,
    Xk: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    W_method: Literal['difference','signed_max'] = 'difference',
    local_grad_method: Literal['auto_diff','bandwidth'] = 'auto_diff',
    fit: bool = True,
    bandwidth: float | None = None,
    exponent: float = 2.0,
    drop_first: bool = True,
    verbose: int = 0
    ) -> np.ndarray:
    """
        :param PredictionModel model: Predictor which can use autodifferentiation
        :param np.ndarray|pd.DataFrame X: Explanatory data
        :param np.ndarray|pd.DataFrame Xk: Knockoff explanatory data
        :param np.ndarray|pd.Series y: Outcome data
        :param Literal['difference','signed_max'] W_method: How to calculate W statistics from importance measures, given the two most common methods.
        :param Literal['auto_diff','bandwidth'] local_grad_method: Method of MALD. Defaults to `'auto_diff'` for exact autodifferentiation. `'bandwidth'` uses the bandwidth approximation when auto differentiation is not available.
        :param bool fit: Whether to fit `model`. Set to `False` if you have already trained it to your satisfaction on the combined explanatory and knockoff data together.
        :param float|None bandwidth: Width if `local_grad_method = 'bandwidth'`
        :param float exponent: Power to take of each MALD value. 1.0 and 2.0 both work reasonably well.
        :param bool drop_first: How to handle one-hot-encoding categorical variables. If `True` the number of associated columns is the number of categories minus 1.
        :param int verbose: How much to print out, for mostly for debugging.
        :returns: W statistics, half the length of `importances`, the same length as the original number of variables
        :rtype: np.ndarray
        
        A one step method of getting W statistics given a `PredictionModel`, wrapping ``importancesFromModel()`` and ``wFromImportances()``
    """
    importances: np.ndarray = importancesFromModel(
        model = model,
        X = X,
        Xk = Xk,
        y = y,
        local_grad_method = local_grad_method,
        fit = fit,
        bandwidth = bandwidth,
        exponent = exponent,
        drop_first = drop_first,
        verbose = verbose
    )
    
    return wFromImportances(
        importances = importances,
        W_method = W_method
    )
#/def wFromModel
