using Pkg
Pkg.add("Flux")
Pkg.add("Images")
Pkg.add("Plots")
Pkg.add("BSON")
Pkg.add("Augmentor")
using Flux
using Images
using Plots
using BSON
using Augmentor

model = Chain(
    # Gate
    Conv((7, 7), 1=>16; bias=false, pad=3, stride=2),
    BatchNorm(16, relu),
    MaxPool((3, 3), pad=1),

    # Layer 1
    Conv((1, 1), 16=>32; bias=false),
    BatchNorm(32, relu),
    Conv((3, 3), 32=>32; bias=false, pad=1, stride=2),
    BatchNorm(32, relu),
    Conv((1, 1), 32=>32; bias=false),
    BatchNorm(32, relu),

    Conv((1, 1), 32=>32; bias=false),
    BatchNorm(32, relu),
    Conv((3, 3), 32=>32; bias=false, pad=1),
    BatchNorm(32, relu),
    Conv((1, 1), 32=>32; bias=false),
    BatchNorm(32, relu),

    # Layer 2
    Conv((1, 1), 32=>64; bias=false),
    BatchNorm(64, relu),
    Conv((3, 3), 64=>64; bias=false, pad=1, stride=2),
    BatchNorm(64, relu),
    Conv((1, 1), 64=>64; bias=false),
    BatchNorm(64, relu),

    Conv((1, 1), 64=>64; bias=false),
    BatchNorm(64, relu),
    Conv((3, 3), 64=>64; bias=false, pad=1),
    BatchNorm(64, relu),
    Conv((1, 1), 64=>64; bias=false),
    BatchNorm(64, relu),

    # Layer 3
    Conv((1, 1), 64=>128; bias=false),
    BatchNorm(128, relu),
    Conv((3, 3), 128=>128; bias=false, pad=1, stride=2),
    BatchNorm(128, relu),
    Conv((1, 1), 128=>128; bias=false),
    BatchNorm(128, relu),

    Conv((1, 1), 128=>128; bias=false),
    BatchNorm(128, relu),
    Conv((3, 3), 128=>128; bias=false, pad=1),
    BatchNorm(128, relu),
    Conv((1, 1), 128=>128; bias=false),
    BatchNorm(128, relu),

    # Classification head
    Flux.flatten,
    Dense(128=>10),
    softmax
)

function createDataset(path, limit)
    X = []
    y = []
    
    labels = readdir(path)
    for label in labels
        println("Generating for ", label)
        files = readdir("$path/$label")
        if limit == true
            files = files[1:2560]
        end
        for file in files
            img = load("$path/$label/$file")
            data = reshape(Float32.(channelview(img)), 28, 28, 1)
            if length(X) == 0
                X = data
            else
                X = cat(X, data, dims=3)
            end
            push!(y, parse(Float32, label))
        end
    end
    return X, y
end

println("Creating train dataset...")
X_train, y_train = createDataset("C:/Users/ddeshler/Downloads/mnist_png/training", true)
println("Creating test dataset...")
X_test, y_test = createDataset("C:/Users/ddeshler/Downloads/mnist_png/testing", false)

X_train = reshape(X_train, 28, 28, 1, :)
X_test = reshape(X_test, 28, 28, 1, :)

data = Flux.DataLoader((X_train, y_train); batchsize=256, shuffle=true)
optimizer = Flux.setup(Adam(), model)

function accuracy()
    correct = 0
    for i in 1:length(y_test)
        probs = model(Flux.unsqueeze(X_test[:, :, :, i], dims=4))
        pred = argmax(probs)[1]-1

        if pred == y_test[i]
            correct += 1
        end
    end
    return correct / length(y_test)
end

function loss(model, x, y)
    return Flux.crossentropy(model(x), Flux.onehotbatch(y, 0:9))
end

# pl = ElasticDistortion(6, 6, sigma=4, scale=0.3, iter=3, border=true) |> SplitChannels() |> PermuteDims((2, 3, 1))
# outbatch(X_train) = Array{Float32}(undef, (28, 28, 1, nobs(X_train)))
# augmentbatch((X_train, y_train)) = (augmentbatch!(outbatch(X_train), X_train, pl), y_train)
# batches = mappedarray(augmentbatch, batchview(X_train, y_train))

best_accuracy = 0
accs = Vector{Float64}()
epochs = 100
for epoch in 1:epochs
    Flux.train!(loss, model, data, optimizer)
    acc = accuracy()
    push!(accs, acc)
    if acc > best_accuracy
        println("New best test accuracy: ", acc)
        global best_accuracy = acc
        BSON.@save "mnist_$epoch.bson" model
    else
        println("Test accuracy: ", acc)
    end
end

plot(range(1, epochs), accs, title="Model Test Accuracy", legend=false)
xlabel!("Epoch")
ylabel!("Accuracy")
savefig("accuracy.png")

println("Finished with best test accuracy: ", best_accuracy)