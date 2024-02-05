using LinearAlgebra
using SparseArrays
using Laplacians
using Random


function getBD(G,S0,S1)
	n=G.n;m=G.m;
	x=Int[];
	y=Int[];
	w=Int[];
	dx=Int[];
	dw=Int[];
	mm=0;
	D=spzeros(n,n)
	for i=1:m
		xx=G.u[i];yy=G.v[i];
		if ((xx==S0) && (yy==S1)) || ((xx==S1) && (yy==S0))
			continue
		end
		if xx!=S0 && xx!=S1 && yy!=S0 && yy!=S1
			mm+=1;
			push!(x,mm);push!(x,mm);
			push!(y,xx);push!(y,yy);
			push!(w,-1);push!(w,1);
			continue
		end
		push!(dx,xx);
		push!(dw,1);
		push!(dx,yy);
		push!(dw,1);
	end
	D=sparsevec(dx,dw,n);
	B=sparse(x,y,w,mm,n);
	return B,D
end


function gre(G ,S0,S1, k ,ep)
	finans=0;
    d0=ep/(3*sqrt(G.n));
	d=4*ep/(3*G.n^2)/(G.n^2);
	t= round(Int64,log(G.n)/(ep^2)/5)+1;
	L = lapsp(G);
	n=G.n;m=G.m;
	V=union(1:n);
	F=setdiff(V,S0,S1)
	B,D=getBD(G,S0,S1); #B mm*n, D n*n
	mm=size(B)[1];
	addlist=Int[];
    rng = MersenneTwister(Int(round(time() * 10000.0)));
	# f = approxchol_sddm(LL, tol=1e-8);
	# firans =sum(f(sel));
	#r2 = randn(rng, nn, 1);
	#r1 = randn(rng, mm, 1);
	S=union(S0,S1);
	x=zeros(n)
	x[S0]=0;x[S1]=1;
	o1=ones(n-2);
	b=L[F,S]*x[S];
	T1=time();
	Dadd=spzeros(n,n);
    for i = 1 : k
        f = approxchol_sddm((L+Dadd)[F,F]);
		h1=zeros(n);h2=zeros(n);
        h1[F]=f(o1);
        h2[F]=f(b);
		Xe = zeros(n);
		Ye = zeros(n);
		z1=zeros(n,1);z2=zeros(n,1);
		dd=D.^(0.5);
		sqrtD=sparse(1:n,1:n,dd);
		rr2 = randn(rng, n-2, t);
		rr1 = randn(rng, mm, t);
        for j = 1 : t
            r2 = rr2[:,j];
			r1 = rr1[:,j];
			xx = B[:,F]'*r1;
			yy = sqrtD[F,F]*r2;
            # f = approxchol_sddm(LL, tol=d);
			z1[S,1].=0;z2[S,1].=0;
			z1[F, 1] = f(xx[:, 1]);
			z2[F, 1] = f(yy[:, 1]);
			for p in F
				Xe[p] += z1[p,1]^2;
				Ye[p] += z2[p,1]^2;
			end
		end
		add = zeros(n);
		for j in F
			add[j] += h1[j]*(1-h2[j])/(1+Xe[j]/t+Ye[j]/t);
		end
		maxx=0;
		bestv=0;
		for j in F
			if (add[j]>maxx)
				maxx = add[j]
				bestv = j
			end
		end
		push!(addlist,bestv)
		D[bestv]+=1;
		Dadd[bestv,bestv]+=1;
		finans += maxx;
	end
	T2=time();
	return addlist;
end
