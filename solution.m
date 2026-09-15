clear; clc; close all;
Nx=400;
Ny=Nx;
point_num=5;


[coord] = coord_2d_irregular(0,1,0,1,Nx,Ny);
[boundary_points] = identifyBoundaryPoints(coord, 0,1,0,1);
[inner_points] = identifyInteriorPoints(coord,0,1,0,1);
coord=[inner_points;boundary_points];
Nb=size(boundary_points,1);
n=size(inner_points,1);
N=n+Nb;

gamma = 1.4;
Ua = zeros(N,4);
u=zeros(N,2);
v=zeros(N,2);
p=zeros(N,2);
rho=zeros(N,2);
E = zeros(N,2);

% [rho(:,1),u(:,1),v(:,1),p(:,1)] = BC_rand_Riemann_a(N,coord);T=0.25;
% [rho(:,1),u(:,1),v(:,1),p(:,1)] = BC_rand_Riemann_b(N,coord);T=0.3;
[rho(:,1),u(:,1),v(:,1),p(:,1)] = BC_rand_Riemann_c(N,coord);T=0.2;
% [rho(:,1),u(:,1),v(:,1),p(:,1)] = BC_rand_Riemann_d(N,coord);T=0.3;

dt=0.1*1e-3;
nt=T/dt; nt=round(nt);

E(:,1)=0.5*(u(:,1).^2+v(:,1).^2)+p(:,1)./(rho(:,1)*(gamma-1));

Ua(:,1) = rho(:,1);
Ua(:,2) = rho(:,1) .* u(:,1);
Ua(:,3) = rho(:,1) .* v(:,1);
Ua(:,4) = rho(:,1) .* E(:,1);

[Dhx_half,Dhy_half,Dhx,Dhy,nearest_indices] = Dh1_rand_2d_half(N,point_num,coord);

vector=zeros(N,point_num-1,2);
for i=1:N
    vector(i,:,1) = coord(nearest_indices(i,2:end), 1) - coord(i, 1);
    vector(i,:,2) = coord(nearest_indices(i,2:end), 2) - coord(i, 2);
end

kdtree = KDTreeSearcher(inner_points);
% x=coord(:,1);y=coord(:,2);
% z1=Ua(:,1,1);
% [X,Y] = meshgrid(min(x):0.02:max(x), min(y):0.02:max(y));
% Z1 = griddata(x, y, z1, X, Y);
% contourf(X, Y,real(Z1), 50); colorbar;

[nearest_indices_bound] = knnsearch(kdtree, boundary_points, 'K', 1);



%% ---- 循环外一次性将固定数据搬到 GPU（单精度）----
Dhx_half_gpu = gpuArray(single(Dhx_half));
Dhy_half_gpu = gpuArray(single(Dhy_half));
Dhx_gpu      = gpuArray(single(Dhx));
Dhy_gpu      = gpuArray(single(Dhy));
nearest_indices_gpu = gpuArray(int32(nearest_indices));   % 索引用 int32
vector_gpu   = gpuArray(single(vector));
nearest_indices_bound=gpuArray(single(nearest_indices_bound));
rho_exact=zeros(N,1);
Ua_gpu = gpuArray(single(Ua));
for k=1:nt

    %% ---- 每步将 Ua 转为 GPU 单精度 ----


    %% ---- 在 GPU 上执行隐式 Rusanov 步 ----
    Ua_gpu = Rusanov_nomesh_limit_implicit( ...
        N, Ua_gpu, Dhx_half_gpu, Dhy_half_gpu, gamma, dt, ...
        nearest_indices_gpu, point_num, vector_gpu, Dhx_gpu, Dhy_gpu);

    Ua_gpu(n+1:N, 1:4) = Ua_gpu(nearest_indices_bound(:,1), 1:4);
    o = (k+1)/(nt+1);
    sprintf('%.2f%%', o * 100)
end

Ua_final   = gather(Ua_gpu);
coord_final = gather(coord);
y = double(coord_final(:,2));
x = double(coord_final(:,1));
z1=double(Ua_final(:,1));
[X,Y] = meshgrid(min(x):0.005:max(x), min(y):0.005:max(y));
Z1 = griddata(x, y, z1, X, Y);
contourf(X, Y,real(Z1), 30, 'LineColor', 'k'); xlabel('x'); ylabel('y'); 
axis equal;colorbar; 



%% =========================================================
function [Dh1x,Dh1y,Dhx,Dhy,nearest_indices] = Dh1_rand_2d_half(N,point_num,coord)
% 导数矩阵计算（CPU双精度）
    Dh1x = zeros(N, point_num);
    Dh1y = zeros(N, point_num);
    Dhx  = zeros(N, point_num);
    Dhy  = zeros(N, point_num);
    n    = point_num;

    kdtree = KDTreeSearcher(coord);
    [nearest_indices, ~] = knnsearch(kdtree, coord, 'K', n);

    P        = zeros(point_num-1, 5);
    hx_need  = zeros(point_num-1, 1);
    hy_need  = zeros(point_num-1, 1);

    % 半距离版本（Dh1x / Dh1y）
    for i = 1:N
        hx_need = (coord(nearest_indices(i,2:n),1) - coord(nearest_indices(i,1),1)) / 2;
        hy_need = (coord(nearest_indices(i,2:n),2) - coord(nearest_indices(i,1),2)) / 2;

        P(:,1) = hx_need;
        P(:,2) = hy_need;
        P(:,3) = 0.5*hx_need.^2;
        P(:,4) = 0.5*hy_need.^2;
        P(:,5) = hx_need.*hy_need;

        W = eye(point_num-1);
        B = zeros(5, point_num);
        for j = 2:point_num-1
            B(:,j) = W(j-1,j-1)*P(j-1,:)';
        end
        B(:,point_num) = W(point_num-1,point_num-1)*P(point_num-1,:)';
        B(:,1) = -sum(B(:,2:point_num), 2);

        A = P'*W*P;
        E = pinv(A)*B;
        Dh1x(i,:) = E(1,:);
        Dh1y(i,:) = E(2,:);
    end

    % 全距离版本（Dhx / Dhy）
    for i = 1:N
        hx_need = coord(nearest_indices(i,2:n),1) - coord(nearest_indices(i,1),1);
        hy_need = coord(nearest_indices(i,2:n),2) - coord(nearest_indices(i,1),2);

        P(:,1) = hx_need;
        P(:,2) = hy_need;
        P(:,3) = 0.5*hx_need.^2;
        P(:,4) = 0.5*hy_need.^2;
        P(:,5) = hx_need.*hy_need;

        W = eye(point_num-1);
        B = zeros(5, point_num);
        for j = 2:point_num-1
            B(:,j) = W(j-1,j-1)*P(j-1,:)';
        end
        B(:,point_num) = W(point_num-1,point_num-1)*P(point_num-1,:)';
        B(:,1) = -sum(B(:,2:point_num), 2);

        A = P'*W*P;
        E = pinv(A)*B;
        Dhx(i,:) = E(1,:);
        Dhy(i,:) = E(2,:);
    end
end


%% =========================================================
function [Ua] = Rusanov_nomesh_limit_implicit( ...
    N, Ua1, Dhx_half, Dhy_half, gamma, dt, nearest_indices, ...
    point_num, vector, Dhx, Dhy)

    % 点隐式线性化格式:
    % (I + dt*J_i) dU_i = -dt * R_i
    % U^{n+1}_i = U^n_i + dU_i

    q  = gpuArray.zeros(1,1, 'single');
    Ua = gpuArray.zeros(N, 4, 'single');
    n_nei = point_num - 1;
    eps_phys = 1e-12;
    eps_lim  = 1e-6;

    %% -------------------- 1) 基本物理量 --------------------
    r  = Ua1(:,1);
    ru = Ua1(:,2);
    rv = Ua1(:,3);
    E  = Ua1(:,4);

    r = max(r, eps_phys);

    u = ru ./ r;
    v = rv ./ r;

    p11 = (gamma-1) .* (E - 0.5 .* r .* (u.^2 + v.^2));
    p11 = max(p11, eps_phys);

    c = sqrt(gamma .* p11 ./ r);
    H = (E + p11) ./ r;

    idx_all = nearest_indices;
    idx_nei = idx_all(:,2:end);

    %% -------------------- 2) 邻域物理量 --------------------
    r_all = r(idx_all);
    u_all = u(idx_all);
    v_all = v(idx_all);
    p_all = p11(idx_all);

    %% -------------------- 3) 梯度 --------------------
    grad_r_x = sum(Dhx .* r_all, 2);
    grad_r_y = sum(Dhy .* r_all, 2);
    grad_u_x = sum(Dhx .* u_all, 2);
    grad_u_y = sum(Dhy .* u_all, 2);
    grad_v_x = sum(Dhx .* v_all, 2);
    grad_v_y = sum(Dhy .* v_all, 2);
    grad_p_x = sum(Dhx .* p_all, 2);
    grad_p_y = sum(Dhy .* p_all, 2);

    grad_r_x_c = repmat(grad_r_x, 1, n_nei);
    grad_r_y_c = repmat(grad_r_y, 1, n_nei);
    grad_u_x_c = repmat(grad_u_x, 1, n_nei);
    grad_u_y_c = repmat(grad_u_y, 1, n_nei);
    grad_v_x_c = repmat(grad_v_x, 1, n_nei);
    grad_v_y_c = repmat(grad_v_y, 1, n_nei);
    grad_p_x_c = repmat(grad_p_x, 1, n_nei);
    grad_p_y_c = repmat(grad_p_y, 1, n_nei);

    grad_r_x_n = grad_r_x(idx_nei);
    grad_r_y_n = grad_r_y(idx_nei);
    grad_u_x_n = grad_u_x(idx_nei);
    grad_u_y_n = grad_u_y(idx_nei);
    grad_v_x_n = grad_v_x(idx_nei);
    grad_v_y_n = grad_v_y(idx_nei);
    grad_p_x_n = grad_p_x(idx_nei);
    grad_p_y_n = grad_p_y(idx_nei);

    %% -------------------- 4) 法向量 --------------------
    nx = vector(:,:,1);
    ny = vector(:,:,2);

    grad_r_dot_n_c = grad_r_x_c .* nx + grad_r_y_c .* ny;
    grad_u_dot_n_c = grad_u_x_c .* nx + grad_u_y_c .* ny;
    grad_v_dot_n_c = grad_v_x_c .* nx + grad_v_y_c .* ny;
    grad_p_dot_n_c = grad_p_x_c .* nx + grad_p_y_c .* ny;

    grad_r_dot_n_n = grad_r_x_n .* nx + grad_r_y_n .* ny;
    grad_u_dot_n_n = grad_u_x_n .* nx + grad_u_y_n .* ny;
    grad_v_dot_n_n = grad_v_x_n .* nx + grad_v_y_n .* ny;
    grad_p_dot_n_n = grad_p_x_n .* nx + grad_p_y_n .* ny;

    %% -------------------- 5) 中心点 / 邻点值 --------------------
    r_c = repmat(r, 1, n_nei);
    u_c = repmat(u, 1, n_nei);
    v_c = repmat(v, 1, n_nei);
    p_c = repmat(p11, 1, n_nei);

    r_n = r(idx_nei);
    u_n = u(idx_nei);
    v_n = v(idx_nei);
    p_n = p11(idx_nei);

    delta_r = r_n - r_c;
    delta_u = u_n - u_c;
    delta_v = v_n - v_c;
    delta_p = p_n - p_c;

    %% -------------------- 6) Van Albada limiter --------------------
    phi_r_c = (grad_r_dot_n_c .* delta_r + abs(grad_r_dot_n_c .* delta_r) + eps_lim) ...
            ./ (grad_r_dot_n_c.^2 + delta_r.^2 + eps_lim);
    phi_r_n = (grad_r_dot_n_n .* delta_r + abs(grad_r_dot_n_n .* delta_r) + eps_lim) ...
            ./ (grad_r_dot_n_n.^2 + delta_r.^2 + eps_lim);

    phi_u_c = (grad_u_dot_n_c .* delta_u + abs(grad_u_dot_n_c .* delta_u) + eps_lim) ...
            ./ (grad_u_dot_n_c.^2 + delta_u.^2 + eps_lim);
    phi_u_n = (grad_u_dot_n_n .* delta_u + abs(grad_u_dot_n_n .* delta_u) + eps_lim) ...
            ./ (grad_u_dot_n_n.^2 + delta_u.^2 + eps_lim);

    phi_v_c = (grad_v_dot_n_c .* delta_v + abs(grad_v_dot_n_c .* delta_v) + eps_lim) ...
            ./ (grad_v_dot_n_c.^2 + delta_v.^2 + eps_lim);
    phi_v_n = (grad_v_dot_n_n .* delta_v + abs(grad_v_dot_n_n .* delta_v) + eps_lim) ...
            ./ (grad_v_dot_n_n.^2 + delta_v.^2 + eps_lim);

    phi_p_c = (grad_p_dot_n_c .* delta_p + abs(grad_p_dot_n_c .* delta_p) + eps_lim) ...
            ./ (grad_p_dot_n_c.^2 + delta_p.^2 + eps_lim);
    phi_p_n = (grad_p_dot_n_n .* delta_p + abs(grad_p_dot_n_n .* delta_p) + eps_lim) ...
            ./ (grad_p_dot_n_n.^2 + delta_p.^2 + eps_lim);

    %% -------------------- 7) 重构 --------------------
    r_c_recon = r_c + 0.5 .* phi_r_c .* grad_r_dot_n_c;
    r_n_recon = r_n - 0.5 .* phi_r_n .* grad_r_dot_n_n;

    u_c_recon = u_c + 0.5 .* phi_u_c .* grad_u_dot_n_c;
    u_n_recon = u_n - 0.5 .* phi_u_n .* grad_u_dot_n_n;

    v_c_recon = v_c + 0.5 .* phi_v_c .* grad_v_dot_n_c;
    v_n_recon = v_n - 0.5 .* phi_v_n .* grad_v_dot_n_n;

    p_c_recon = p_c + 0.5 .* phi_p_c .* grad_p_dot_n_c;
    p_n_recon = p_n - 0.5 .* phi_p_n .* grad_p_dot_n_n;

    r_c_recon = max(r_c_recon, eps_phys);
    r_n_recon = max(r_n_recon, eps_phys);
    p_c_recon = max(p_c_recon, eps_phys);
    p_n_recon = max(p_n_recon, eps_phys);

    %% -------------------- 8) X方向Rusanov通量 --------------------
    mask_x = nx >= 0;

    rL = gpuArray.zeros(N, n_nei, 'single');
    rR = gpuArray.zeros(N, n_nei, 'single');
    uL = gpuArray.zeros(N, n_nei, 'single');
    uR = gpuArray.zeros(N, n_nei, 'single');
    vL = gpuArray.zeros(N, n_nei, 'single');
    vR = gpuArray.zeros(N, n_nei, 'single');
    pL = gpuArray.zeros(N, n_nei, 'single');
    pR = gpuArray.zeros(N, n_nei, 'single');

    rL_y = gpuArray.zeros(N, n_nei, 'single');
    rR_y = gpuArray.zeros(N, n_nei, 'single');
    uL_y = gpuArray.zeros(N, n_nei, 'single');
    uR_y = gpuArray.zeros(N, n_nei, 'single');
    vL_y = gpuArray.zeros(N, n_nei, 'single');
    vR_y = gpuArray.zeros(N, n_nei, 'single');
    pL_y = gpuArray.zeros(N, n_nei, 'single');
    pR_y = gpuArray.zeros(N, n_nei, 'single');

    LHS = gpuArray.zeros(4, 4, N, 'single');

    rL(mask_x) = r_c_recon(mask_x);   rR(mask_x) = r_n_recon(mask_x);
    uL(mask_x) = u_c_recon(mask_x);   uR(mask_x) = u_n_recon(mask_x);
    vL(mask_x) = v_c_recon(mask_x);   vR(mask_x) = v_n_recon(mask_x);
    pL(mask_x) = p_c_recon(mask_x);   pR(mask_x) = p_n_recon(mask_x);

    rL(~mask_x) = r_n_recon(~mask_x); rR(~mask_x) = r_c_recon(~mask_x);
    uL(~mask_x) = u_n_recon(~mask_x); uR(~mask_x) = u_c_recon(~mask_x);
    vL(~mask_x) = v_n_recon(~mask_x); vR(~mask_x) = v_c_recon(~mask_x);
    pL(~mask_x) = p_n_recon(~mask_x); pR(~mask_x) = p_c_recon(~mask_x);

    rL = max(rL, eps_phys); rR = max(rR, eps_phys);
    pL = max(pL, eps_phys); pR = max(pR, eps_phys);

    HL = (gamma .* pL) ./ ((gamma-1) .* rL) + 0.5 .* (uL.^2 + vL.^2);
    HR = (gamma .* pR) ./ ((gamma-1) .* rR) + 0.5 .* (uR.^2 + vR.^2);
    aL = sqrt(gamma .* pL ./ rL);
    aR = sqrt(gamma .* pR ./ rR);

    WL = cat(3, rL, rL.*uL, rL.*vL, rL.*HL - pL);
    WR = cat(3, rR, rR.*uR, rR.*vR, rR.*HR - pR);

    fL = cat(3, rL.*uL, rL.*uL.*uL + pL, rL.*uL.*vL, rL.*uL.*HL);
    fR = cat(3, rR.*uR, rR.*uR.*uR + pR, rR.*uR.*vR, rR.*uR.*HR);

    smax = max(abs(uL)+aL, abs(uR)+aR);
    flux_x = 0.5 .* (fR + fL + repmat(smax,1,1,4) .* (WL - WR));

    %% -------------------- 9) Y方向Rusanov通量 --------------------
    mask_y = ny >= 0;

    rL_y(mask_y) = r_c_recon(mask_y);   rR_y(mask_y) = r_n_recon(mask_y);
    uL_y(mask_y) = u_c_recon(mask_y);   uR_y(mask_y) = u_n_recon(mask_y);
    vL_y(mask_y) = v_c_recon(mask_y);   vR_y(mask_y) = v_n_recon(mask_y);
    pL_y(mask_y) = p_c_recon(mask_y);   pR_y(mask_y) = p_n_recon(mask_y);

    rL_y(~mask_y) = r_n_recon(~mask_y); rR_y(~mask_y) = r_c_recon(~mask_y);
    uL_y(~mask_y) = u_n_recon(~mask_y); uR_y(~mask_y) = u_c_recon(~mask_y);
    vL_y(~mask_y) = v_n_recon(~mask_y); vR_y(~mask_y) = v_c_recon(~mask_y);
    pL_y(~mask_y) = p_n_recon(~mask_y); pR_y(~mask_y) = p_c_recon(~mask_y);

    rL_y = max(rL_y, eps_phys); rR_y = max(rR_y, eps_phys);
    pL_y = max(pL_y, eps_phys); pR_y = max(pR_y, eps_phys);

    HL_y = (gamma .* pL_y) ./ ((gamma-1) .* rL_y) + 0.5 .* (uL_y.^2 + vL_y.^2);
    HR_y = (gamma .* pR_y) ./ ((gamma-1) .* rR_y) + 0.5 .* (uR_y.^2 + vR_y.^2);
    aL_y = sqrt(gamma .* pL_y ./ rL_y);
    aR_y = sqrt(gamma .* pR_y ./ rR_y);

    WL_y = cat(3, rL_y, rL_y.*uL_y, rL_y.*vL_y, rL_y.*HL_y - pL_y);
    WR_y = cat(3, rR_y, rR_y.*uR_y, rR_y.*vR_y, rR_y.*HR_y - pR_y);

    gL = cat(3, rL_y.*vL_y, rL_y.*uL_y.*vL_y, rL_y.*vL_y.*vL_y + pL_y, rL_y.*vL_y.*HL_y);
    gR = cat(3, rR_y.*vR_y, rR_y.*uR_y.*vR_y, rR_y.*vR_y.*vR_y + pR_y, rR_y.*vR_y.*HR_y);

    smax2 = max(abs(vL_y)+aL_y, abs(vR_y)+aR_y);
    flux_y = 0.5 .* (gR + gL + repmat(smax2,1,1,4) .* (WL_y - WR_y));

    %% -------------------- 10) 中心通量 --------------------
    Hc = (gamma .* p11) ./ ((gamma-1) .* r) + 0.5 .* (u.^2 + v.^2);
    f_center = [r.*u, r.*u.*u + p11, r.*u.*v, r.*u.*Hc];
    g_center = [r.*v, r.*u.*v, r.*v.*v + p11, r.*v.*Hc];

    %% -------------------- 11) 空间残差 --------------------
    Dhx_face = Dhx_half(:,2:end);
    Dhy_face = Dhy_half(:,2:end);
    Dhx1 = Dhx_half(:,1);
    Dhy1 = Dhy_half(:,1);

    flux_x_sum = reshape(sum(Dhx_face .* flux_x, 2), [N,4]);
    flux_y_sum = reshape(sum(Dhy_face .* flux_y, 2), [N,4]);
    center_term = Dhx1 .* f_center + Dhy1 .* g_center;

    Res = flux_x_sum + flux_y_sum + center_term;   % N x 4

    dt_used = dt;

    %% -------------------- 13) 局部雅可比近似 J_i --------------------
    ax = sum(abs(Dhx_half), 2);
    ay = sum(abs(Dhy_half), 2);
    beta = ax .* (abs(u) + c) + ay .* (abs(v) + c);

    [Ablock, Bblock] = localEulerJacobiansGPU_safe(r,u,v,H,gamma);

    ax3    = reshape(ax,      1, 1, N);
    ay3    = reshape(ay,      1, 1, N);
    alpha3 = reshape(1 + dt_used .* beta, 1, 1, N);

    LHS = dt_used .* (Ablock .* ax3 + Bblock .* ay3);

    LHS(1,1,:) = LHS(1,1,:) + alpha3;
    LHS(2,2,:) = LHS(2,2,:) + alpha3;
    LHS(3,3,:) = LHS(3,3,:) + alpha3;
    LHS(4,4,:) = LHS(4,4,:) + alpha3;

    %% -------------------- 14) 右端项并逐点求解 --------------------
    rhs = -bsxfun(@times, dt_used, Res);   % N x 4
    rhs3 = reshape(rhs.', [4,1,N]);

    dU3 = pagefun(@mldivide, LHS, rhs3);   % 4 x 1 x N
    dU  = reshape(permute(dU3,[3,1,2]), [N,4]);

    %% -------------------- 15) 更新 --------------------
    Ua = Ua1 + dU;

    Ua(:,1) = max(Ua(:,1), eps_phys);

    rho_new = Ua(:,1);
    u_new = Ua(:,2) ./ rho_new;
    v_new = Ua(:,3) ./ rho_new;
    Emin = 0.5 .* rho_new .* (u_new.^2 + v_new.^2) + eps_phys/(gamma-1);
    Ua(:,4) = max(Ua(:,4), Emin);

    %% -------------------- 16) 收敛指标 --------------------
    q = max(abs(dU(:,1)));
end


%% =========================================================
function [Ablock,Bblock] = localEulerJacobiansGPU_safe(r,u,v,H,gamma)
    N = numel(r);
    q2 = u.^2 + v.^2;

    Ablock = gpuArray.zeros(4, 4, N, 'single');
    Bblock = gpuArray.zeros(4, 4, N, 'single');

    % ---------- A = dF/dU ----------
    Ablock(1,1,:) = 0;
    Ablock(1,2,:) = 1;
    Ablock(1,3,:) = 0;
    Ablock(1,4,:) = 0;

    Ablock(2,1,:) = 0.5*(gamma-3).*u.^2 + 0.5*(gamma-1).*v.^2;
    Ablock(2,2,:) = (3-gamma).*u;
    Ablock(2,3,:) = -(gamma-1).*v;
    Ablock(2,4,:) = gamma-1;

    Ablock(3,1,:) = -u.*v;
    Ablock(3,2,:) = v;
    Ablock(3,3,:) = u;
    Ablock(3,4,:) = 0;

    Ablock(4,1,:) = -u.*H + 0.5*(gamma-1).*u.*q2;
    Ablock(4,2,:) = H - (gamma-1).*u.^2;
    Ablock(4,3,:) = -(gamma-1).*u.*v;
    Ablock(4,4,:) = gamma.*u;

    % ---------- B = dG/dU ----------
    Bblock(1,1,:) = 0;
    Bblock(1,2,:) = 0;
    Bblock(1,3,:) = 1;
    Bblock(1,4,:) = 0;

    Bblock(2,1,:) = -u.*v;
    Bblock(2,2,:) = v;
    Bblock(2,3,:) = u;
    Bblock(2,4,:) = 0;

    Bblock(3,1,:) = 0.5*(gamma-1).*u.^2 + 0.5*(gamma-3).*v.^2;
    Bblock(3,2,:) = -(gamma-1).*u;
    Bblock(3,3,:) = (3-gamma).*v;
    Bblock(3,4,:) = gamma-1;

    Bblock(4,1,:) = -v.*H + 0.5*(gamma-1).*v.*q2;
    Bblock(4,2,:) = -(gamma-1).*u.*v;
    Bblock(4,3,:) = H - (gamma-1).*v.^2;
    Bblock(4,4,:) = gamma.*v;
end
