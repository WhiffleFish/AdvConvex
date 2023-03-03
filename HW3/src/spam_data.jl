const SPAM_URL = "https://github.com/probml/pmtk3/tree/master/data/spamData"
const SPAM_URL_RAW = "https://raw.githubusercontent.com/probml/pmtk3/master/data/spamData/spam.data"

function get_spam_data()
    mat = readdlm(Downloads.download(SPAM_URL_RAW))
    m = Matrix(transpose(mat))
    Y = @view(m[end, :])
    map!(Y,Y) do y_i # change y_i ∈ {1,0} to {1, -1}
        iszero(y_i) ? -1 : y_i
    end
    return m
end

function train_test_split(mat::Matrix, test_pct::Float64; shuffle=true, transform=true)
    all_idxs = collect(1:size(mat,2))
    shuffle && shuffle!(all_idxs)
    split_thresh = floor(Int, length(all_idxs) * test_pct)

    train_idxs = all_idxs[split_thresh+1 : end]
    test_idxs = all_idxs[1:split_thresh]

    A_train = mat[:, train_idxs]
    X_train = log.(A_train[1:end-1,:] .+ 0.1)
    Y_train = A_train[end, :]

    A_test = mat[:, test_idxs]
    X_test = log.(A_test[1:end-1, :] .+ 0.1)
    Y_test = A_test[end, :]
    return X_train, Y_train, X_test, Y_test
end
