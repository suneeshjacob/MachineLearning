def viterbi(T,E,p,s):
    h = []
    p_c = p.copy()
    v_c = np.multiply(p_c,E[:,s[0]])
    h.append(np.argmax(v_c))
    for i in s[1:]:
        t_c = np.multiply((T*E[:,i]).T,v_c)
        v_c = np.array([np.max(j) for j in t_c])
        h.append(np.argmax(v_c))
    return h
