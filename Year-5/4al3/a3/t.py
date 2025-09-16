def stochastic_gradient_descent(self):
    prev_loss = float('inf')
    train_losses, test_losses, plot_train_losses, plot_test_losses = [], [], [], []
    samples = 0
    stopping_epoch = 0
    found_optimal = False

    # execute the stochastic gradient descent function for defined epochs
    for epoch in range(self.epoch):

        # shuffle to prevent repeating update cycles
        features, output = shuffle(self.X_train, self.Y_train)

        for i, feature in enumerate(features):
            gradient = self.compute_gradient(feature, output[i])
            self.weights = self.weights - (self.learning_rate * gradient)

        # Part 1
        train_loss = self.compute_loss(self.X_train, self.Y_train)
        test_loss = self.compute_loss(self.X_test, self.Y_test)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if epoch % (self.epoch / 10) == 0 and epoch != self.epoch:
            plot_train_losses.append(train_loss)
            plot_test_losses.append(test_loss)

        if epoch % 1000 == 0:
            print(f"Epoch is: {epoch} and Loss is : {train_loss}")

        if abs(prev_loss - train_loss) < self.loss_threshold and not found_optimal:
            print(f"Early stopping at epoch {epoch + 1} due to minimal loss improvement.")
            stopping_epoch = epoch
            found_optimal = True
            self.best_weights = self.weights
        prev_loss = train_loss

        # print("The minimum number of iterations taken are:",stopping_epoch)

        # check for convergence - end

        # below code will be required for Part 3

        # Part 3
        samples += 1

    print("Training ended...")
    print("Final weights are: {}".format(self.weights))
    print("Early stopped weights are: {}".format(self.best_weights))
    print("The minimum number of iterations taken are:", stopping_epoch)

    plot_X = []
    for i in range(10):
        plot_X.append(i * (self.epoch / 10))
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))
    ax[0].plot(plot_X, plot_train_losses)
    ax[1].plot(plot_X, plot_test_losses)
    fig.tight_layout()

    # below code will be required for Part 3
    print("The minimum number of samples used are:", samples)