from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, BasicIterativeMethod, DeepFool, SaliencyMapMethod
from art.attacks.evasion import DeepFool, HopSkipJump, NewtonFool, ElasticNet
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import CarliniL2Method
import numpy as np 


def _ensure_numpy(x):
    if hasattr(x, "values"):
        return x.values
    return x

def fgsm_attack(classifier,X_scaled_test, eps): 
    # Generate adversarial examples using FGSM
    attack = FastGradientMethod(estimator=classifier, eps=eps)
    X_test_adv = attack.generate(x=_ensure_numpy(X_scaled_test))
    return X_test_adv

def pgd_attack(classifier, X_scaled_test, eps):
    attack = ProjectedGradientDescent(estimator=classifier, eps=eps)
    X_test_adv = attack.generate(x=_ensure_numpy(X_scaled_test))
    return X_test_adv

def bim_attack(classifier, X_scaled_test, eps):
    attack = BasicIterativeMethod(estimator=classifier, eps=eps)
    X_test_adv = attack.generate(x=_ensure_numpy(X_scaled_test))
    return X_test_adv

def jsma_attack(classifier, X_scaled_test, theta, gamma):
    attack = SaliencyMapMethod(classifier=classifier,theta=theta, gamma=gamma)
    X_test_adv = attack.generate(x=_ensure_numpy(X_scaled_test))
    return X_test_adv

def cw_attack(classifier, X_scaled_test):
    attack = CarliniL2Method(classifier=classifier, confidence=0.0,
    targeted=False,
    learning_rate=0.001,
    binary_search_steps=10,
    max_iter=20,
    initial_const=0.9,
    batch_size=1,)
    
    # Generate adversarial examples
    X_test_adv = attack.generate(x=_ensure_numpy(X_scaled_test))
    
    return X_test_adv

def deepfool_attack(classifier, X_scaled_test): 
    attack = DeepFool(classifier = classifier, max_iter= 10, epsilon=1e-6)
    x_test_adv = attack.generate(x = _ensure_numpy(X_scaled_test))
    return x_test_adv

def hopskipjump_attack(classifier, X_scaled_test): 
    attack = HopSkipJump(classifier=classifier, norm = 'inf', max_iter= 10, max_eval= 100, init_eval= 10)
    x_test_adv = attack.generate(x = _ensure_numpy(X_scaled_test))
    return x_test_adv

def newtonfool_attack(classifier, X_scaled_test): 
    attack = NewtonFool(classifier=classifier, max_iter= 100, eta = 0.001)
    x_test_adv = attack.generate(x = _ensure_numpy(X_scaled_test))
    return x_test_adv

def elasticnet_attack(classifier, X_scaled_test): 
    attack = ElasticNet(classifier=classifier, confidence= 0.0, targeted= False, max_iter= 10, binary_search_steps= 10, learning_rate= 0.01, initial_const=0.01)
    x_test_adv = attack.generate(x = _ensure_numpy(X_scaled_test))
    return x_test_adv


