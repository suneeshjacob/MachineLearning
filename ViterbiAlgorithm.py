def viterbi(T, E, p, s):
    v = p * E[:, s[0]]
    psi = []
    for i in s[1:]:
        scores = v[:, None] * T
        psi.append(np.argmax(scores, axis=0))
        v = np.max(scores, axis=0) * E[:, i]

    h = [0 for i in range(len(s))]
    h[-1] = np.argmax(v)
    for t in range(len(psi)-1, -1, -1):
        h[t] = psi[t][h[t+1]]
    return h
