#include("Graph.jl")
#include("Matrix.jl")

using LinearAlgebra
using SparseArrays
using Laplacians

function getResult(G, ff, k, S, s)
    sf = getSf(G.n, k, s, ff)
    ss = getSs(G.n, k, s, ff)
    Omg = getOmegaS(G, k, ff, S)
    Afs = getAfsS(G, k, ff, S)
    of = ones(G.n-k)
    ans = of' * Omg * (Afs * ss + sf)
    return ans
end

function randomSelect( k2)
    S =union(1:k2);
    for i = 1 : k2
        #push!(S, abs(rand(Int)%(G.n-G.n0-G.n1))+G.n0+G.n1+1)
        S[i]=G.n0+G.n1+i;
        #S[i]=abs(rand(Int)%(G.n))+1;
    end
    return S
end

function topDegree(G, k2)
    S=union(1:1);setdiff!(S,1);
    d = zeros(G.n)
    for i=1: G.m
        nu = G.u[i]
        nv = G.v[i]
        d[nu] += 1
        d[nv] += 1
    end
    sev = G.n-G.n0-G.n1
    cho = union(G.n0+G.n1+1:G.n);
    dd=d;
    for rep = 1 : k2
        #dd = d;
        xx = argmax(dd[cho])+G.n0+G.n1;
        #setdiff!(cho,xx);
        dd[xx]=-100;
        push!(S, xx)
    end
    return S
end

function PageRankCentrality(G)
    n=G.n;
    a=adjsp(G)
    l=lapsp(G)
    d=zeros(n)
    for i=1:n
        d[i]=l[i,i]
    end
    p=d.^(-1)'*a
    dd=0.85;

    v=rand(n)
    v=v./norm(v,1)
    for i=1:10
        v=dd*ones(n)+(1-dd)*p*v
    end
    return v
end

function BetweennessCentrality(G)
    gg = zeros(Int, G.n)
    foreach(i -> gg[i] = i, 1 : G.n)
    g = Array{Array{Int32, 1}, 1}(undef, G.n)
    foreach(i -> g[i] = [], 1 : G.n)
    for i=1:G.m
        u=G.u[i];
        v=G.v[i];
        push!(g[u], v)
        push!(g[v], u)
    end
    C = zeros(G.n)
    p = Array{Array{Int32, 1}, 1}(undef, G.n)
    d = zeros(Int32, G.n)
    S = zeros(Int32, G.n+10)
    sigma = zeros(G.n)
    Q = zeros(Int32, G.n+10)
    delta = zeros(G.n)
    for s = 1 : G.n
        foreach(i -> p[i] = [], 1 : G.n)
        top = 0
        sigma .= 0
        sigma[s] = 1.0
        d .= -1
        d[s] = 0
        front = 1
        rear = 1
        Q[1] = s

        while front <= rear
            v = Q[front]
            front += 1
            top += 1
            S[top] = v
            for w in g[v]
                if d[w] < 0
                    rear += 1
                    Q[rear] = w
                    d[w] = d[v] + 1
                end
                if d[w] == (d[v] + 1)
                    sigma[w] += sigma[v]
                    push!(p[w], v)
                end
            end
        end

        delta .= 0

        while top > 0
            w = S[top]
            top -= 1
            for v in p[w]
                delta[v] += ((sigma[v] / sigma[w]) * (1 + delta[w]))
                if w != s
                    C[w] += delta[w]
                end
            end
        end

    end

    return C
end

function topBetweenness(G,  k2)
    S = union(1:1);
    setdiff!(S,1);
    sev = G.n-G.n0-G.n1
    cho = union(G.n0+G.n1+1:G.n);
    for rep = 1 : k2
        dd = ClosenessCentrality(G, S)
        xx = argmax(dd[cho])+G.n0+G.n1;
        #setdiff!(cho,xx);
        dd[xx]=-100;
        push!(S, xx)
    end
    return S
end


function topPageRank(G, k2, Ev)
    S = union(1:1);
    setdiff!(S,1);
    sev = G.n-G.n0-G.n1
    cho = union(G.n0+G.n1+1:G.n);
dd = PageRank(G, S)
    for rep = 1 : k2
        #dd = PageRank(G, S)
        xx = argmax(dd[cho])+G.n0+G.n1;
        #setdiff!(cho,xx);
        dd[xx]=-100;
        push!(S, xx)
    end
    return S
end

function ClosenessCentrality(G)
    gg = zeros(Int, G.n)
    foreach(i -> gg[i] = i, 1 : G.n)
    g = Array{Array{Int32, 1}, 1}(undef, G.n)
    foreach(i -> g[i] = [], 1 : G.n)
    for i=1:G.m
        u=G.u[i];
        v=G.v[i];
        push!(g[u], v)
        push!(g[v], u)
    end
    C = zeros(G.n)
    d = zeros(Int32, G.n)
    Q = zeros(Int32, G.n+10)
    for s = 1 : G.n
        d .= -1
        d[s] = 0
        front = 1
        rear = 1
        Q[1] = s

        while front <= rear
            v = Q[front]
            front += 1
            for w in g[v]
                if d[w] < 0
                    rear += 1
                    Q[rear] = w
                    d[w] = d[v] + 1
                end
            end
        end

        C[s] = sum(d)
    end

    foreach(i -> C[i] = 1.0 / C[i], 1 : G.n)

    return C
end

function topCloseness(G,  k2)
    S = union(1:1);
    setdiff!(S,1);
    sev = G.n-G.n0-G.n1
    cho = union(G.n0+G.n1+1:G.n);
    dd = ClosenessCentrality(G, S)
    for rep = 1 : k2
        #dd = ClosenessCentrality(G, S)
        xx = argmax(dd[cho])+G.n0+G.n1;
        dd[xx]=-100;
        #setdiff!(cho,xx);
        push!(S, xx)
    end
    return S
end

function calcc(G,S)
    h=G.n0+G.n1;
    xs=ones(h);
    for i=1:G.n0
        xs[i]=0;
    end
    L=lap(G);
    for i in S
        L[i,i]+=1;
        L[G.n0+1,G.n0+1]+=1;
        L[i,G.n0+1]-=1;
        L[G.n0+1,i]-=1;
        #b[i-h]-=1;
    end
    sel = zeros(G.n - G.n0- G.n1);
    for i = G.n0+G.n1+1 : G.n
        sel[i-(G.n0+G.n1)] =  sum(L[i,1:G.n0]) - sum(L[i,1:G.n0+G.n1]);
    end
    Linv = inv(L[G.n0+G.n1+1:G.n,G.n0+G.n1+1:G.n]);
	firans = sum(Linv * sel);
    return firans;
end
