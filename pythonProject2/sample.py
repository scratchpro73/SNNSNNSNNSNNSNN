import numpy as np
import matplotlib.pyplot as plt
# import scipy.linalg as slin
# import networkx as nx
import tensorflow as tf

# import warnings
# warnings.filterwarnings('ignore')

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

NETWORK_SHAPE = [28, 16, 10]
LEARNING = True
batch_labels = []


class Connection:
    def __init__(self, from_depth, from_number, to_depth, to_number):
        self.from_depth = from_depth
        self.from_number = from_number
        self.to_depth = to_depth
        self.to_number = to_number

        self.aggrivation = 0

        self.weight = np.random.randint(10, 1000) / 50

    def sigmoid(self, x):
        ex = np.exp(x)
        return ex / (1 + ex)

    def hebbian_update(self, pre_neuron, post_neuron, learning_rate=0.001):
        self.aggrivation = 0
        if pre_neuron.activated and post_neuron.activated:
            self.aggrivation = 1
            self.weight += learning_rate * self.sigmoid(np.abs(self.weight) - 10)
        elif pre_neuron.activated and post_neuron.activated == False:
            self.aggrivation = -1
            if self.weight > 0.1:
                self.weight -= learning_rate * self.sigmoid(np.abs(self.weight) - 10) * 0.5
        elif pre_neuron.activated == False and post_neuron.activated:
            self.aggrivation = -1
            if self.weight > 0.1:
                self.weight -= learning_rate * self.sigmoid(np.abs(self.weight) - 10) * 0.5
                # else:
        #     if self.weight > 3 or self.weight < 1 :        #         self.weight = 0.05 + self.weight * 0.95 #weight decay for preventing ultra high weight


class Neuron:

    # high level functions use multiple low level functions following the corresponding sequence of level order

    def __init__(self, depth, number):  # input has depth 0, INITIALIZATION, level 0
        self.initial_pressure = 0
        self.pressure = 0
        self.speed = 0
        self.depth = depth
        self.number = number
        self.threshold = 10
        self.threshold_min = 10
        self.activated = False
        self.signal = 0

    def singular_read__from(self, other):  # INTERNAL FUNCTION, level 1
        connected_neurons = np.sum([NETWORK_SHAPE[d] for d in range(0, self.depth)])
        return other.signal / connected_neurons

    def singular_update__to_from_with(self, other, connection_obj):  # INTERNAL FUNCTION, level 2
        self.speed += self.singular_read__from(other) * connection_obj.weight

    def full_update__to(self, N, C):  # EXTERNAL FUNCTION, level 3
        for other_n in N:
            for other_ne in other_n:
                if other_ne != self:
                    if other_ne.depth < self.depth:
                        connection = C[self.depth][self.number][other_ne.depth][other_ne.number]
                        self.singular_update__to_from_with(other_ne, connection)
                        if LEARNING == True:
                            connection.hebbian_update(other_ne, self)

                            # After this for loop, the neuron has updated its speed
        self.pressure += self.speed  # This represents a signal

        self.speed *= 0.5  # Friction
        self.speed -= self.pressure * 0.1

        if self.pressure > self.threshold:
            self.speed += 1
            self.threshold += 5
            self.activated = True
            self.signal = 3
        else:
            self.signal = 0
            self.activated = False

            # self.threshold *= 0.5
        self.threshold += (self.threshold_min - self.threshold) * 0.1

    def reset_pressure(self):
        """Reset the pressure of the neuron to its initial value."""
        self.pressure = self.initial_pressure
        self.threshold = self.threshold_min


N = []
for depth in range(np.size(NETWORK_SHAPE)):
    N.append([Neuron(number=0, depth=depth)])
    for number in range(NETWORK_SHAPE[depth] - 1):
        N[depth].append(Neuron(number=number + 1, depth=depth))
    N[depth] = np.array(N[depth], dtype=object)
N = np.array(N, dtype=object)

# N is the array of neurons with rows for depth, cols for number (neuron id)

C = [[[[None for _ in range(NETWORK_SHAPE[to_depth])] for to_depth in range(len(NETWORK_SHAPE))]
      for _ in range(NETWORK_SHAPE[from_depth])] for from_depth in range(len(NETWORK_SHAPE))]

for from_depth in range(len(N)):
    for from_number in range(len(N[from_depth])):
        for to_depth in range(len(N)):
            for to_number in range(len(N[to_depth])):
                if from_depth + from_number * 100 != to_depth + to_number * 100:
                    C[from_depth][from_number][to_depth][to_number] = Connection(
                        from_depth=from_depth, from_number=from_number,
                        to_depth=to_depth, to_number=to_number)
C = np.array(C, dtype=object)


# C is the 4D array of connetcions with depth and numbers


# Visualization function
def visualize_network(N, C, ax):
    ax.clear

    # Draw neurons
    for depth in range(len(N)):
        for number in range(len(N[depth])):
            x = depth
            y = -number
            target = N[depth][number]

            if target.activated == True:
                color = 'red'
            else:
                color = 'lightblue'

            if target != None:
                ax.scatter(x, y, s=(np.abs(target.pressure) + 1) * 100, color=color)
                ax.text(x, y, f"{depth}-{number}", fontsize=9, ha='right')

                # Draw connections
    # for from_depth in range(len(C)):
    #     for from_number in range(len(C[from_depth])):
    #         for to_depth in range(len(C[from_depth][from_number])):
    #             for to_number in range(len(C[from_depth][from_number][to_depth])):
    #                 targ_object = C[from_depth][from_number][to_depth][to_number]
    #                 if targ_object is not None:
    #                     from_x, from_y = from_depth, -from_number
    #                     to_x, to_y = to_depth, -to_number
    #                     if targ_object.aggrivation == 1:
    #                         col = 'red'
    #                     elif targ_object.aggrivation == -1:
    #                         col = 'blue'
    #                     else:
    #                         col = 'gray'
    #                     ax.plot([from_x, to_x], [from_y, to_y], col, linestyle='-', linewidth=targ_object.weight)

                        # Visualize the network


INF = 10 ** 10

epochs = 28  # The width of the image
batches = 100  # Number of mnist datas

# VISUALIZATION INTERVAL FOR TRAINING
interval = INF  # INF for performance, Erase connection visualization for fast visuals

output_neuron_log_pressure = [[] for _ in range(NETWORK_SHAPE[-1])]
output_neuron_log_threshold = [[] for _ in range(NETWORK_SHAPE[-1])]

fig, ax = plt.subplots(figsize=(6, 4))
ax.set_title("Neural Network Visualization")
ax.set_xlabel("Depth (Layers)")
ax.set_ylabel("Neuron Number")
ax.invert_yaxis()

# train and visualize
for id in range(batches):
    print(f">> {id + 1} batches out of {batches} -----------------------")
    input_log = np.multiply(x_train[id], 0.01)

    # Log the label for the current batch
    batch_labels.append(y_train[id])

    for epoch in range(epochs):
        if (epoch + 1) % interval == 0:
            print(f">> {epoch + 1} out of {epochs} epochs <<")
            visualize_network(N, C, ax)
            plt.pause(0.001)
            plt.cla()
            ax.invert_yaxis()

        for n in N:
            for ne in n:
                ne.full_update__to(N, C)

        for i in range(28):
            N[0][i].speed += input_log[i][epoch]

        for i in range(NETWORK_SHAPE[-1]):
            output_neuron_log_pressure[i].append(N[-1][i].pressure)
            output_neuron_log_threshold[i].append(N[-1][i].threshold)

    for depth in range(len(N)):
        for number in range(len(N[depth])):
            N[depth][number].reset_pressure()

plt.close(fig)  # Close the figure window after the loop

print(y_train[0:batches:1])

# plt.figure(figsize=(10, 6))
# # Plot the pressure logs of all output neurons
# for i, log in enumerate(output_neuron_log_pressure):
#     plt.plot(np.divide(np.array(range(len(log))),epochs), log, label=f'Output Neuron Pressure {i + 1}')
# for i, log in enumerate(output_neuron_log_threshold):
#     plt.plot(np.divide(np.array(range(len(log))),epochs), log, label=f'Output Neuron Threshold {i + 1}')
#
# # Annotate with labels
# for i, label in enumerate(batch_labels):
#     plt.axvline(x=i, color='gray', linestyle='--', alpha=0.5)
#     plt.text(i, max(max(log) for log in output_neuron_log_pressure + output_neuron_log_threshold),
#              f'Label: {label}', rotation=90, fontsize=8, va='bottom')
#
# plt.title('Pressure Logs of All Output Neurons')
# plt.xlabel('Batches')
# plt.ylabel('Pressure / Threshold')
# plt.legend()
# plt.show()


LEARNING = False

###################################################################
###################################################################
###################################################################
# TESTING AREA
###################################################################
###################################################################
###################################################################
batch_labels = []

epochs = 28  # The width of the image
batches = 20  # Number of mnist data

# VISUALIZATION INTERVAL FOR TESTING
interval = 7  # INF for performance, Erase connection visualization for fast visuals

output_neuron_log_pressure = [[] for _ in range(NETWORK_SHAPE[-1])]
output_neuron_log_threshold = [[] for _ in range(NETWORK_SHAPE[-1])]

fig, ax = plt.subplots(figsize=(6, 4))
ax.set_title("Neural Network Visualization")
ax.set_xlabel("Depth (Layers)")
ax.set_ylabel("Neuron Number")
ax.invert_yaxis()

# train and visualize
for id in range(batches):
    print(f">> {id + 1} batches out of {batches} -----------------------")
    input_log = np.multiply(x_test[id], 0.01)

    # Log the label for the current batch
    batch_labels.append(y_test[id])

    for epoch in range(epochs):
        if (epoch + 1) % interval == 0:
            print(f">> {epoch + 1} out of {epochs} epochs <<")
            visualize_network(N, C, ax)
            plt.pause(0.001)
            plt.cla()
            ax.invert_yaxis()

        for n in N:
            for ne in n:
                ne.full_update__to(N, C)

        for i in range(28):
            N[0][i].speed += input_log[i][epoch]

        for i in range(NETWORK_SHAPE[-1]):
            output_neuron_log_pressure[i].append(N[-1][i].pressure)
            output_neuron_log_threshold[i].append(N[-1][i].threshold)

    for depth in range(len(N)):
        for number in range(len(N[depth])):
            N[depth][number].reset_pressure()

plt.close(fig)  # Close the figure window after the loop

print(y_test[0:batches:1])

plt.figure(figsize=(10, 6))
# Plot the pressure logs of all output neurons
for i, log in enumerate(output_neuron_log_pressure):
    plt.plot(np.divide(np.array(range(len(log))), epochs), log)  # , label=f'Output Neuron Pressure {i + 1}')
for i, log in enumerate(output_neuron_log_threshold):
    plt.plot(np.divide(np.array(range(len(log))), epochs), log)  # , label=f'Output Neuron Threshold {i + 1}')

# Annotate with labels
for i, label in enumerate(batch_labels):
    plt.axvline(x=i, color='gray', linestyle='--', alpha=0.5)
    plt.text(i, max(max(log) for log in output_neuron_log_pressure + output_neuron_log_threshold),
             f'Label: {label}', rotation=90, fontsize=8, va='bottom')

plt.title('Pressure Logs of All Output Neurons')
plt.xlabel('Batches')
plt.ylabel('Pressure / Threshold')
plt.legend()
plt.show()