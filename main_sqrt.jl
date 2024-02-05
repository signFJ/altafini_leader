include("graph.jl")
include("core.jl")
include("schur.jl")
include("edgecore.jl")

using LinearAlgebra
using Laplacians

fname = open("filename.txt", "r")
str   = readline(fname);
nn     = parse(Int, str);

for nnnn=1:nn
#######S0 \cup S1

str = readline(fname);
str = split(str);
G   = get_graph(str[1]);
on=G.n;om=G.m;
Gc=findconnect(G);
G=Gc;
n=G.n;m=G.m;

# beta=0.05;

#p=Int(round(log(n)/eps1^2/500));
#pp=Int(round(p^0.5));
#p=50; ###sqrtn
#pp=20; ####random walks
### 0.05 0.03 0.01
p1=400;
pp1=60;
p2=120;
pp2=45;
p3=50;
pp3=30;
# p1=p3;pp1=pp3;
### 0.3 0.2


#len=Int(round(1/beta^2));
len=1000;
selc_num=50;
kmax=30;

fans1=open("undirect_ans.txt","a");
println(fans1,str[1],' ',on,' ',om,' ',n,' ',m)

println(fans1,"eps(kdd)=0.3,0.1,eps(new)=0.05,0.03,0.01,p(sqrt)=pp(sample)=,",p1,' ',p2,' ',p3,' ',pp1,' ',pp2,' ',pp3," len=",len,",selc_num=",selc_num,",kmax=",kmax);
close(fans1)

################Init
d=zeros(n);
for i=1:n
    d[i]=size(G.nbr[i])[1]
end
S=Int[];
for i=1:selc_num
    push!(S,argmax(d))
    d[argmax(d)]=0;
end
#S=rand(1:n,50)
#S=union!(S)
# S=S[1:2*selc_num];
G=get_graph_union(G,S)
n=G.n;m=G.m;
d=zeros(n);
for i=1:n
    d[i]=size(G.nbr[i])[1]
end
d[argmax(d)]=0;
S=Int[];
for i=1:selc_num
    push!(S,argmax(d))
    d[argmax(d)]=0;
end
#S=rand(1:n,50)
#S=union!(S)
# S=S[1:selc_num];
G=get_graph_union(G,S)
n=G.n;m=G.m;
#L=lapsp(G);
#a=adjsp(G);
d=zeros(n);
for i=1:n
    d[i]=size(G.nbr[i])[1]
end
S0=argmax(d);
d[argmax(d)]=0;
S1=argmax(d);
for i=1:n
    d[i]=size(G.nbr[i])[1]
end
############init_end

####kdd method
t0=time()
# ans_kdd1=gre(G ,S0,S1, kmax ,0.3);
println(111)
println(111)
t1=time()
# ans_kdd2=gre(G ,S0,S1, kmax ,0.1);
println(222)
t2=time()
##########exact


##########sample walks
###10
L=lapsp(G);
ans_sqrt1=Int[];
d=zeros(n);
for i=1:n
    d[i]=size(G.nbr[i])[1]
end
t3=time()
for ttt=1:kmax
    W=init_sqrt(G,S0,S1,p1,pp1,len);
    sco=upd_sqrt(W,d,n);
    xx=argmax(sco);
    push!(G.nbr[xx],S1);push!(G.nbr[S1],xx);
    push!(ans_sqrt1,xx);
    d[xx]+=1;
end
t4=time()
for i=1:length(kmax)
    pop!(G.nbr[ans_sqrt1[i]]);
    pop!(G.nbr[S1]);
end
println(333)
####2
ans_sqrt2=Int[];
d=zeros(n);
for i=1:n
    d[i]=size(G.nbr[i])[1]
end
t5=time()
for ttt=1:kmax
    W=init_sqrt(G,S0,S1,p2,pp2,len);
    sco=upd_sqrt(W,d,n);
    xx=argmax(sco);
    push!(G.nbr[xx],S1);push!(G.nbr[S1],xx);
    push!(ans_sqrt2,xx);
    d[xx]+=1;
end
t6=time()
for i=1:length(kmax)
    pop!(G.nbr[ans_sqrt2[i]]);
    pop!(G.nbr[S1]);
end
println(444)
###3
# ans_sqrt3=Int[];
# d=zeros(n);
# for i=1:n
#     d[i]=size(G.nbr[i])[1]
# end
# t7=time()
# for ttt=1:kmax
#     W=init_sqrt(G,S0,S1,p3,pp3,len);
#     sco=upd_sqrt(W,d,n);
#     xx=argmax(sco);
#     push!(G.nbr[xx],S1);push!(G.nbr[S1],xx);
#     push!(ans_sqrt3,xx);
#     d[xx]+=1;
# end
# t8=time()
# for i=1:length(kmax)
#     pop!(G.nbr[ans_sqrt3[i]]);
#     pop!(G.nbr[S1]);
# end
# println(555)
##################
# L=lapsp(G);
fans1=open("undirect_ans.txt","a");
println(fans1,"kdd,sqrt,exa")
# println(fans1,"time: ",t1-t0,' ',t2-t1,' ',t4-t3,' ',t6-t5,' ',t8-t7)
# println(fans1,"orig opinion: ",getans(G,S0,S1,[]))
println(fans1,"time: ",t4-t3,' ',t6-t5)
for i=1:kmax
    # println(fans1,getans(G,S0,S1,ans_kdd[1:i]),' ',getans(G,S0,S1,ans_sqrt[1:i]) ,' ',getans(G,S0,S1,ans_exa[1:i]))
    # println(fans1,getans(G,S0,S1,ans_kdd1[1:i],L),' ',getans(G,S0,S1,ans_kdd2[1:i],L),' ',getans(G,S0,S1,ans_sqrt1[1:i],L),' ',getans(G,S0,S1,ans_sqrt2[1:i],L),' ',getans(G,S0,S1,ans_sqrt3[1:i],L))
    println(fans1,getans(G,S0,S1,ans_sqrt1[1:i],L),' ',getans(G,S0,S1,ans_sqrt2[1:i],L))
end
println(fans1)
close(fans1)

end

close(fname)
