""" Image Classification using Transfer Learning
	with light weight MobileNet
"""
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2


def define_model(input_shape, n_classes):
	""" Stack layers on pretrained model
	"""
	inp = tf.keras.layers.Input(shape=X_train.shape[1:])

	mobileNet = MobileNetV2(include_top=False)
	for layer in mobileNet.layers:
		layer.trainable = False

	mbnetOut = mobileNet(inp)
	avgPool = tf.keras.layers.GlobalAveragePooling2D()(mbnetOut)
	dense = tf.keras.layers.Dense(512, activation=tf.nn.relu)(avgPool)
	out = tf.keras.layers.Dense(n_classes)(dense)
	model = tf.keras.Model(inp, out)

	return model


# Train
epochs = 10
batch_size = 32
num_classes = 10

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train = X_train / 255
X_test = X_test / 255

model = define_model(X_train.shape[1:], num_classes)
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam(0.001)
accuracy = tf.metrics.Accuracy()
step = tf.Variable(1, name="global_step")


@tf.function
def train_step(inputs, labels):
	with tf.GradientTape() as tape:
		logits = model(inputs)
		loss_value = loss(labels, logits)

	gradients = tape.gradient(loss_value, model.trainable_variables)
	optimizer.apply_gradients(
		zip(gradients, model.trainable_variables))
	step.assign_add(1)
	accuracy_value = accuracy(labels, tf.argmax(logits, -1))

	return loss_value, accuracy_value


num_batches = X_train.shape[0] // batch_size

ckpt = tf.train.Checkpoint(model=model,
				optimizer=optimizer, step=step)
manager = tf.train.CheckpointManager(
	ckpt, "log/checkpoints", max_to_keep=2)

writer = tf.summary.create_file_writer("log/TransferLearning")
with writer.as_default():
	for epoch in range(epochs):
		for batch in range(num_batches):
			start_idx = epoch * batch_size
			end_idx = (epoch + 1) * batch_size
			X_batch = X_train[start_idx:end_idx]
			Y_batch = Y_train[start_idx:end_idx]

			loss_value, accuracy_value = train_step(
				X_batch, Y_batch)

			if batch % 10 == 0:
				# Log accuracy and loss
				tf.summary.scalar("accuracy", 
					accuracy_value, step.numpy())
				tf.summary.scalar("loss", 
					loss_value, step.numpy())

		# Local log accuracy and loss on train and 
		# validation data at end of epoch
		print("Epoch: ", epoch + 1)

		ckpt_path = manager.save()
		print("Saved checkpoint at: ", ckpt_path)

		start_idx = 0
		end_idx = 128

		labels = Y_train[start_idx:end_idx]
		logits = model(X_train[start_idx:end_idx])
		accuracy_value = accuracy(labels, tf.argmax(logits, -1))
		loss_value =loss(labels, logits)

		print("Train Loss: {1}, Train Accuracy: {2}"
				.format(epoch + 1, loss_value, accuracy_value))

		labels = Y_test[start_idx:end_idx]
		logits = model(X_test[start_idx:end_idx])
		accuracy_value = accuracy(labels, tf.argmax(logits, -1))
		loss_value = loss(labels, logits)

		print("Validation Loss: {1}, Validation Accuracy: {2}"
				.format(epoch + 1, loss_value, accuracy_value))
