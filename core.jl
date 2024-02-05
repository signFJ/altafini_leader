using LinearAlgebra
using SparseArrays
using Laplacians
using Random
# using Arpack

function PageRankCentrality(G)
    n=G.n;
    a=adjsp(G)
    l=lapsp(G)
    d=zeros(n,n)
    for i=1:n
        d[i,i]=l[i,i]
    end
    p=a*inv(d)
    dd=0.15;

    v=rand(n)
    v=v./norm(v,1)
    for i=1:10
        v=dd/n*ones(n)+(1-dd)*p*v
		v=v./norm(v,1)
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

function getans(G,S0,S1,xx,L)
	n=G.n
	V=union(1:n)
	S=union(S0,S1)
	F=setdiff(V,S)
	x=zeros(n)
	x[S1]=1;
	#L=lapsp(G);
	len=size(xx)[1];
	for i=1:len
		L[xx[i],S1]-=1;L[S1,xx[i]]-=1;
		L[xx[i],xx[i]]+=1;L[S1,S1]+=1;
	end
	# L[xx,S1].-=1;
	# L[S1,xx].-=1;
	# L[xx,xx].+=1;
	# L[S1,S1]+=size(xx)[1];
	f=approxchol_sddm(L[F,F])
	aa=-sum(f(L[F,S]*x[S]));
	for i=1:len
		L[xx[i],S1]+=1;L[S1,xx[i]]+=1;
		L[xx[i],xx[i]]-=1;L[S1,S1]-=1;
	end
	return aa;
end

function getans_direct(G,S0,S1,xx)
	n=G.n
	V=union(1:n)
	S=union(S0,S1)
	F=setdiff(V,S)
	x=zeros(n)
	x[S1]=1;
	L=lapsp_direct(G);
	len=size(xx)[1];
	for i=1:len
		L[xx[i],S1]-=1;
		L[xx[i],xx[i]]+=1;
	end
	LL=zeros(n,n);
	LL.=L;
	# L[xx,S1].-=1;
	# L[S1,xx].-=1;
	# L[xx,xx].+=1;
	# L[S1,S1]+=size(xx)[1];
	return -sum(inv(LL[F,F])*L[F,S]*x[S]);
end

function qpl(G,S0,k);
	s=zeros(k);
	kk=1;
	for i=1:k
		if kk in S0
			kk+=1;
		end
		s[i]=kk;
		kk+=1
	end
	tmp=getans(G,S0,S1);
	indx=k;
	while s[1]<=n-k+1
		tans=getans(L,n,s,k);
		if tans>tmp
			tmp=tans;
		end
		s[k]+=1;
		if s[k]>n
			s[k]-=1;
			while (indx>=1) && (s[indx]==n+indx-k)
				indx-=1;
				if indx==0
					indx=1;
					break;
				end
			end
			s[indx]+=1;
			for i=indx+1:k
				s[i]=s[i-1]+1;
			end
			indx=k;
		end
	end
	return tmp;
end


function eigcal(a::AbstractArray; nev = 1, tol = 0.0)
    f = approxchol_sddm(a)
    op = Laplacians.SqLinOp(true,1.0,size(a,1),f)
    e = eigs(op, which=:LM, nev=nev, tol=tol)
    e[1] .= 1 ./ e[1]
    return e
end

function lan(Lgg,n,t,err)

	u=ones(n-t);
	v=ones(n-t);
	ff = approxchol_sddm(Lgg, tol=err);
	delt=1;
    while delt>1e-3
        u=v;
        v=ff(u);
        u=u/norm(u);
        v=v/norm(v);
        delt=abs((u'*ff(u))/(u'*ff(u))-(u'*ff(u))/(v'*ff(v)));
		#delt=norm(u-v);
    end
	return 1/(ff(v)'*v);
end



function getW(G)
	return sparse(1:G.m,1:G.m,G.w)
end


function getB(G)
	v1=zeros(2*G.m)
	v2=zeros(2*G.m)
	v1[1:G.m]=1:G.m;v1[G.m+1:2*G.m]=1:G.m;
	v2[1:G.m]=G.u;v2[G.m+1:2*G.m]=G.v;
	o=zeros(2*G.m)
	o[1:G.m].=1;o[G.m+1:2*G.m].=-1;
	return sparse(v1,v2,o)
end

function adjsp_direct(G)
	return sparse(G.u,G.v,ones(G.m),n,n)
end

function adjsp(G :: Graph)
	uu=zeros(2*G.m);
	vv=zeros(2*G.m);
	ww=zeros(2*G.m);
	uu[1:G.m]=G.u;uu[G.m+1:2*G.m]=G.v;
	vv[1:G.m]=G.v;vv[G.m+1:2*G.m]=G.u;
	ww[1:2*G.m].=1;
	return sparse(uu,vv,ww)
end

