from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
import keras.backend as K

from sklearn.model_selection import train_test_split


dataset = loadtxt('C:\\Users\\DELL\\Documents\\Projet matlab\\dataset\\OSI_codagSimplePy.txt', delimiter=',')
# dataframe=pandas.read_csv('dataset.csv',delimiter=',',header=1)
# dataset=dataframe.values
input = dataset[:, 0:17]
target = dataset[:, 17:18]
target2 = loadtxt('C:\\Users\\DELL\\Documents\\Projet matlab\\dataset\\codageTarget2.txt', delimiter=',')
targets = [target ,target2]


def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
    return;


def create_baseline(encode, network_topology,input_activation_function, output_activation_function, optimiseur, perte, nb, input, target):

    model = Sequential()

    model.add(Dense(network_topology[0], input_dim=17, activation=input_activation_function))
    for neural in range(1, len(network_topology)) :
        model.add(Dense(neural, activation=input_activation_function))

    model.add(Dense(encode, activation=output_activation_function))

    model.compile(loss=perte, optimizer=optimiseur, metrics=['acc'])
    print(model.summary())
    plot_model(model, to_file='model_plot.png', show_shapes='true', show_layer_names='true')
    print(encode, input_activation_function, output_activation_function, optimiseur, 1)
    hist = model.fit(input, target, epochs=120, batch_size=40, verbose=1)
    scores = model.evaluate(i_test, t_test, verbose=0)
    print(model.metrics_names)
    best_scores = scores
    print(scores)
    for i in range(1, nb):
        print(encode, input_activation_function, output_activation_function, optimiseur, i + 1)
        reset_weights(model)
        hist = model.fit(input, target, epochs=120, batch_size=40, verbose=1)
        scores = model.evaluate(i_test, t_test, verbose=0)
        print(scores)
        if ((scores[0] < best_scores[0]) and (scores[1] > best_scores[1])):
            print("better model")
            best_scores = scores

    return best_scores

encodage = 1
best_loss = 1
best_acc = 0
best_encode = 1
best_optimiseur = 'rmsprop'
best_output_function = 'sigmoid'
best_input_function = 'relu'

network_topology ={"T1": [[20],[35]],
                   "T2": [[10 ,8],[20 , 8],[32 , 24]] ,
                   "T3": [[10,8,4],[20,8,4],[32 ,24,12]]
                   }
output_activation_function = ['sigmoid', 'hard_sigmoid']
input_activation_function = ['relu', 'tanh', 'sigmoid']
optimisers = ['rmsprop', 'adam', 'adagrad', 'adadelta']
for t in targets:
    print("\n Encodage = " + str(encodage))
    for ti in network_topology :
        for netTopo in network_topology[ti]:
            for input_function in input_activation_function:
                for output_function in output_activation_function:
                    for optimiseur in optimisers:
                        i_train, i_test, t_train, t_test = train_test_split(input, t, test_size=0.40, random_state=42)
                        scores = create_baseline(encodage, netTopo,input_function, output_function, optimiseur, 'mse', 5,
                                                 i_train,
                                                 t_train)
                        if ((scores[0] < best_loss) and (scores[1] > best_acc)):
                            best_acc = scores[1]
                            best_loss = scores[0]
                            best_encode = encodage
                            best_optimiseur = optimiseur
                            best_input_function = input_function
                            best_output_function = output_function

    encodage = encodage + 1
print(best_loss, best_acc, best_encode, best_input_function, best_output_function, best_optimiseur)

# model=create_baseline(t_train,'relu','adam','mse')
# hist=model.fit(i_train, t_train, epochs=1000, batch_size=150,verbose=1,validation_data=(X_test, Y_test))
# mse_value, mae_value = model.evaluate(i_test, t_test, verbose=0)
# print(r2_score(t_train, t_test))



