from HandGestureDataset import HandGestureDataset, joint_names
import matplotlib.pyplot as plt
import numpy as np

dataset = HandGestureDataset()
dataset = dataset.load_dataset()
print(len(dataset))


def plot_joint_distribution(data, joint_idx):
    # Get all samples for the given joint index
    samples = [sample for sample, label in data]
    joint_samples = [sample[joint_idx] for sample in samples]

    # Get all unique labels in the dataset
    labels = list(set([label for sample, label in data]))

    # Set up color map for each label
    color_map = plt.get_cmap('hsv', len(labels))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each sample with a different color based on label
    for i, label in enumerate(labels):
        label_samples = [joint_samples[j] for j in range(len(joint_samples)) if data[j][1] == label]
        xs = np.array(label_samples)[:, 0]
        ys = np.array(label_samples)[:, 1]
        zs = np.array(label_samples)[:, 2]
        ax.scatter(xs, ys, zs, c=[color_map(i)], label=label)

    ax.legend()
    ax.set_title(f"Joint {joint_names[joint_idx]} distribution")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


# Plot distribution for each joint
for i in range(21):
    plot_joint_distribution(dataset, i)
