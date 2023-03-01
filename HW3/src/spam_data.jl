const SPAM_URL = "https://github.com/probml/pmtk3/tree/master/data/spamData"
const SPAM_URL_RAW = "https://raw.githubusercontent.com/probml/pmtk3/master/data/spamData/spam.data"

function get_spam_data()
    mat = readdlm(Downloads.download(SPAM_URL_RAW))
    return Matrix(transpose(mat))
end


function train_test_split(mat::Matrix, test_pct::Float64; shuffle=true)
    all_idxs = collect(1:size(mat,2))
    shuffle && shuffle!(all_idxs)
    split_thresh = floor(Int, length(all_idxs) * test_pct)
    A_train = mat[:, split_thresh+1 : end]

    X_train = A_train[1:end-1,:]
    Y_train = A_train[end, :]
    A_test = mat[:, 1:split_thresh]
    X_test = A_test[1:end-1, :]
    Y_test = A_test[end, :]
    return X_train, Y_train, X_test, Y_test
end
