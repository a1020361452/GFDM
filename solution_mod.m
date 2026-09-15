clear; clc; close all;
warning off all

%% ===================== 0. 基础参数 =====================
point_num = 9;
gamma = 1.4;
CFL = 1.0;                % 隐式格式CFL数（点隐式可取较大值）

% 读取数据
m1 = textread('all_point.txt', '%f');
m2 = textread('L.txt', '%f');
m3 = textread('face.txt', '%f');
m4 = textread('R.txt', '%f');

%% ===================== 1. 坐标处理 =====================
m = size(m1,1);
coord = zeros(m/3,2);
for i = 1:m/3
    coord(i,1) = m1((i-1)*3+1,1);
    coord(i,2) = m1((i-1)*3+2,1);
end
     % plot(coord(:,1), coord(:,2), 'b.', 'MarkerSize', 8);  % 内部点（蓝色）

m = size(m2,1);
boundary_points_outside_L = zeros(m/3,2);
for i = 1:m/3
    boundary_points_outside_L(i,1) = m2((i-1)*3+1,1);
    boundary_points_outside_L(i,2) = m2((i-1)*3+2,1);
end

m = size(m3,1);
boundary_points_inter = zeros(m/3,2);
for i = 1:m/3
    boundary_points_inter(i,1) = m3((i-1)*3+1,1);
    boundary_points_inter(i,2) = m3((i-1)*3+2,1);
end
% 剔除翼型尖端点（如 (0.6,0.2) 和 (0.6,0)）
idx = boundary_points_inter(:,1) == 0.6 & boundary_points_inter(:,2) == 0.2;
boundary_points_inter(idx,:) = [];
idx = boundary_points_inter(:,1) == 0.6 & boundary_points_inter(:,2) == 0;
boundary_points_inter(idx,:) = [];

m = size(m4,1);
boundary_points_outside_R = zeros(m/3,2);
for i = 1:m/3
    boundary_points_outside_R(i,1) = m4((i-1)*3+1,1);
    boundary_points_outside_R(i,2) = m4((i-1)*3+2,1);
end
idx = boundary_points_outside_R(:,1) == 0.6 & boundary_points_outside_R(:,2) == 0;
boundary_points_outside_R(idx,:) = [];

% 合并所有边界点，用于内部点筛选
del_mat = [boundary_points_outside_L; boundary_points_inter; boundary_points_outside_R];
del_mat = unique(del_mat, 'rows');
tol = 1e-4;
idx = ~ismembertol(coord, del_mat, tol, 'ByRows', true);
inner_points = coord(idx, :);
idx = inner_points(:,1) == 0.6 & inner_points(:,2) == 0;
inner_points(idx,:) = [];

% 最终坐标：内部点 + 三类边界点
coord = [inner_points; boundary_points_outside_L; boundary_points_outside_R; boundary_points_inter];

% 点数统计
NbL  = size(boundary_points_outside_L,1);
NbR  = size(boundary_points_outside_R,1);
Nbin = size(boundary_points_inter,1);
n    = size(inner_points,1);
N    = n + NbL + NbR + Nbin;

%% ===================== 2. 边界索引处理 =====================
% （KDTree 操作仍在CPU上完成）
kdtree = KDTreeSearcher(inner_points);
nearest_indices_L0 = knnsearch(kdtree, boundary_points_outside_R, 'K', 1);
nearest_indices_L1 = knnsearch(kdtree, boundary_points_inter, 'K', 1);
i_out = nearest_indices_L1;                  % 用于内部边界处理

%% ===================== 3. 内部边界法向量计算 =====================
% （CPU上计算，结果后续转为GPU单精度）
kdtree_inter = KDTreeSearcher(boundary_points_inter);
[norm_near] = knnsearch(kdtree_inter, boundary_points_inter, 'K', 3);

nx = zeros(Nbin,1);
ny = zeros(Nbin,1);
for i = 1:Nbin
    P = boundary_points_inter(norm_near(i,1),:);
    A = boundary_points_inter(norm_near(i,2),:);
    B = boundary_points_inter(norm_near(i,3),:);
    vec_PA = A - P;
    vec_PB = B - P;
    tangent = vec_PB - vec_PA;
    normal_ccw = [-tangent(2), tangent(1)];
    normal_cw  = [ tangent(2), -tangent(1)];
    ref_vec = P - [2, 0.1];
    dot_ccw = dot(normal_ccw, ref_vec);
    dot_cw  = dot(normal_cw, ref_vec);
    outer_normal = normal_ccw .* (dot_ccw > dot_cw) + normal_cw .* (dot_ccw <= dot_cw);
    unit_outer_normal = outer_normal / norm(outer_normal);
    nx(i) = unit_outer_normal(1);
    ny(i) = unit_outer_normal(2);
end
% 手动修正部分法向（根据物理规律）
nx(1:601) = 0;   ny(1:601) = -1;
nx(640)   = -1;  ny(640)   = 0;
nx(641:760) = 0;  ny(641:760) = 1;
nx(761)   = -1/sqrt(2);
ny(761)   = 1/sqrt(2);

%% ===================== 4. 导数矩阵与邻近索引 =====================
% （CPU上计算，结果后续转为GPU单精度）
[Dhx_half, Dhy_half, Dhx, Dhy, nearest_indices] = Dh1_rand_2d_half(N, point_num, coord);

%% ===================== 5. 邻近向量计算 =====================
% （利用邻近点坐标构造方向向量）
[~, K] = size(nearest_indices);  % K = point_num
M = K - 1;                       % 邻居数量
vector = zeros(N, M, 2);
idx_center = nearest_indices(:, 1);
idx_neigh  = nearest_indices(:, 2:end);
cx = coord(:, 1);  cy = coord(:, 2);
cx_center = cx(idx_center);
cy_center = cy(idx_center);
cx_neigh = cx(idx_neigh);
cy_neigh = cy(idx_neigh);
cx_center_rep = repmat(cx_center, 1, M);
cy_center_rep = repmat(cy_center, 1, M);
vector(:,:,1) = cx_neigh - cx_center_rep;
vector(:,:,2) = cy_neigh - cy_center_rep;

%% ===================== 6. 全部转为GPU单精度 =====================
% 初始物理量（定义为single，之后直接用于GPU数组的赋值）
r0  = single(1);
u0  = single(3);                % Ma=3, 方向沿x轴
v0  = single(0);
E0  = single(1/(gamma*(gamma-1)) + 0.5*3^2);

% 边界索引（gpuArray，整型）
L_rows1    = gpuArray(int32(n+1 : n+NbL));
R_rows1    = gpuArray(int32(n+NbL+1 : n+NbL+NbR));
inner_rows = gpuArray(int32(n+NbL+NbR+1 : N));
i_out      = gpuArray(int32(i_out));
nearest_indices_L0 = gpuArray(int32(nearest_indices_L0));

% 导数矩阵、向量、邻近索引 转为 GPU 单精度
Dhx_half = gpuArray(single(Dhx_half));
Dhy_half = gpuArray(single(Dhy_half));
Dhx      = gpuArray(single(Dhx));
Dhy      = gpuArray(single(Dhy));
nearest_indices = gpuArray(int32(nearest_indices));
vector   = gpuArray(single(vector));

% 法向量转为 GPU 单精度
nx_gpu = gpuArray(single(nx));
ny_gpu = gpuArray(single(ny));

% 初始守恒变量 Ua = [rho, rho*u, rho*v, rho*E]
Ua = gpuArray.zeros(N, 4, 'single');
Ua(:,1) = r0;
Ua(:,2) = r0 * u0;
Ua(:,3) = r0 * v0;
Ua(:,4) = r0 * E0;

%% ===================== 7. 时间推进参数 =====================
T  = 4;
dt = single(0.1e-3);
nt = round(T / dt);
k  = 0;
RHS = 1;

%% ===================== 8. 主迭代循环 =====================
while (RHS > 1e-8) && (k < nt)
    k = k + 1;
    tic;

    % 备份当前步
    Ua_old = Ua;

    % ---------- 隐式求解 ----------
    [Ua] = Rusanov_nomesh_limit_implicit( ...
        N, Ua, Dhx_half, Dhy_half, gamma, dt, nearest_indices, ...
        point_num, vector, Dhx, Dhy, CFL);

    % ---------- 边界条件 ----------
    % 入口（左侧来流）
    Ua(L_rows1, 1) = r0;
    Ua(L_rows1, 2) = r0 * u0;
    Ua(L_rows1, 3) = r0 * v0;
    Ua(L_rows1, 4) = r0 * E0;  % 注意此处能量修正
    % 出口（外推）
    Ua(R_rows1, :) = Ua(nearest_indices_L0, :);
    
    % 物面（无穿透）
    rho_wall = Ua(i_out, 1);
    u_inner  = Ua(i_out, 2) ./ rho_wall;
    v_inner  = Ua(i_out, 3) ./ rho_wall;
    p_wall   = (gamma-1) * (Ua(i_out,4) - 0.5*rho_wall.*(u_inner.^2 + v_inner.^2));
    
    temp   = nx_gpu .* u_inner + ny_gpu .* v_inner;
    ub_new = u_inner - nx_gpu .* temp;
    vb_new = v_inner - ny_gpu .* temp;
    
    Ua(inner_rows, 1) = rho_wall;
    Ua(inner_rows, 2) = rho_wall .* ub_new;
    Ua(inner_rows, 3) = rho_wall .* vb_new;
    Ua(inner_rows, 4) = p_wall/(gamma-1) + 0.5*(Ua(inner_rows,2).^2 + Ua(inner_rows,3).^2) ./ Ua(inner_rows,1);

    % ---------- 收敛判断 ----------
    RHS = max(max(abs(Ua - Ua_old)));   % 使用密度变化更为合理，这里保留原写法

    % 进度输出
    progress = k / nt * 100;
    step_time = toc;
    fprintf('进度：%.2f%%  | 剩余时间：%.2fH  | RHS：%.2e\n', ...
            progress, step_time * (nt - k) / 3600, RHS);
end

%% ===================== 9. 后处理（转回CPU double） =====================
Ua_final = gather(double(Ua));
fprintf('迭代完成，结果已转回CPU。\n');

% 计算导出量
x = coord(:,1);
y = coord(:,2);
r = Ua_final(:,1);
u = Ua_final(:,2)./r;
v = Ua_final(:,3)./r;
p = (gamma-1) * (Ua_final(:,4) - 0.5*r.*(u.^2 + v.^2));

% 绘制压力云图
[X, Y] = meshgrid(0:0.005:3, 0:0.005:1);
Z1 = griddata(x, y, p, X, Y);

% 构造遮挡区域（翼型内部）
x_airfoil = linspace(0.6, 3, 241)';
y_airfoil = linspace(0, 0.2, 21)';
x_upper = x_airfoil;          y_upper = 0.2*ones(241,1);
x_lower = flip(x_airfoil);    y_lower = zeros(241,1);
x_L = 0.6*ones(21,1);         y_L = y_airfoil;
x_R = 3*ones(21,1);           y_R = flip(y_airfoil);
x_poly = [x_upper; x_R; x_lower; x_L];
y_poly = [y_upper; y_R; y_lower; y_L];
in_airfoil = inpolygon(X, Y, x_poly, y_poly);
Z1(in_airfoil) = NaN;

figure(1);
contourf(X, Y, real(Z1), 30, 'LineColor', 'k');
xlabel('x'); ylabel('y'); colorbar; axis equal;
title('压力场');

% ====================================================================
%                            函数定义部分
% ====================================================================

function [Dh1x, Dhy1x, Dhx, Dhy, nearest_indices] = Dh1_rand_2d_half(N, point_num, coord)
    % 生成半离散导数矩阵（二阶精度，含二阶项）
    Dh1x = zeros(N, point_num);
    Dhy1x = zeros(N, point_num);
    Dhx  = zeros(N, point_num);
    Dhy  = zeros(N, point_num);
    W = eye(point_num-1);
    n = point_num;

    kdtree = KDTreeSearcher(coord);
    [nearest_indices, nearest_distances] = knnsearch(kdtree, coord, 'K', n);

    P = zeros(point_num-1,5);
    for i = 1:N
        % ---- 半差 ----
        hx_half = (coord(nearest_indices(i,2:n),1) - coord(nearest_indices(i,1),1))/2;
        hy_half = (coord(nearest_indices(i,2:n),2) - coord(nearest_indices(i,1),2))/2;
        P(:,1) = hx_half; P(:,2) = hy_half;
        P(:,3) = 0.5*hx_half.^2; P(:,4) = 0.5*hy_half.^2; P(:,5) = hx_half.*hy_half;
        B = zeros(5, point_num);
        for j = 2:point_num-1
            B(:,j) = W(j-1,j-1)*(P(j-1,:)');
        end
        B(:,point_num) = W(point_num-1,point_num-1)*(P(point_num-1,:)');
        B(:,1) = -sum(B(:,2:point_num),2);
        A = P'*W*P;
        E = pinv(A)*B;
        Dh1x(i,:) = E(1,:);
        Dhy1x(i,:) = E(2,:);

        % ---- 全差 ----
        hx_full = coord(nearest_indices(i,2:n),1) - coord(nearest_indices(i,1),1);
        hy_full = coord(nearest_indices(i,2:n),2) - coord(nearest_indices(i,1),2);
        P(:,1) = hx_full; P(:,2) = hy_full;
        P(:,3) = 0.5*hx_full.^2; P(:,4) = 0.5*hy_full.^2; P(:,5) = hx_full.*hy_full;
        A = P'*W*P;
        E = pinv(A)*B;
        Dhx(i,:) = E(1,:);
        Dhy(i,:) = E(2,:);
    end
end

function [Ua, q, dt_used] = Rusanov_nomesh_limit_implicit( ...
    N, Ua1, Dhx_half, Dhy_half, gamma, dt, nearest_indices, ...
    point_num, vector, Dhx, Dhy, CFL)
    % 点隐式线性化格式:
    % (I + dt*J_i) dU_i = -dt * R_i
    % U^{n+1}_i = U^n_i + dU_i

    q  = gpuArray.zeros(1,1, 'single');
    Ua = gpuArray.zeros(N, 4, 'single');
    n_nei = point_num - 1;
    eps_phys = single(1e-12);
    eps_lim  = single(1e-6);

    %% -------------------- 1) 基本物理量 --------------------
    r  = Ua1(:,1);
    ru = Ua1(:,2);
    rv = Ua1(:,3);
    E  = Ua1(:,4);
    r = max(r, eps_phys);

    u = ru ./ r;
    v = rv ./ r;

    p11 = max((gamma-1) .* (E - 0.5 .* r .* (u.^2 + v.^2)), eps_phys);
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
    nx = vector(:,:,1)/2;
    ny = vector(:,:,2)/2;

    grad_r_dot_n_c = grad_r_x_c .* nx + grad_r_y_c .* ny;
    grad_u_dot_n_c = grad_u_x_c .* nx + grad_u_y_c .* ny;
    grad_v_dot_n_c = grad_v_x_c .* nx + grad_v_y_c .* ny;
    grad_p_dot_n_c = grad_p_x_c .* nx + grad_p_y_c .* ny;

    grad_r_dot_n_n = grad_r_x_n .* nx + grad_r_y_n .* ny;
    grad_u_dot_n_n = grad_u_x_n .* nx + grad_u_y_n .* ny;
    grad_v_dot_n_n = grad_v_x_n .* nx + grad_v_y_n .* ny;
    grad_p_dot_n_n = grad_p_x_n .* nx + grad_p_y_n .* ny;

    %% -------------------- 5) 差值与限制器 --------------------
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

    %% -------------------- 6) 重构 --------------------
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

    %% -------------------- 7) X方向 Rusanov 通量 --------------------
    mask_x = nx >= 0;
    rL = gpuArray.zeros(N, n_nei, 'single');
    rR = gpuArray.zeros(N, n_nei, 'single');
    uL = gpuArray.zeros(N, n_nei, 'single');
    uR = gpuArray.zeros(N, n_nei, 'single');
    vL = gpuArray.zeros(N, n_nei, 'single');
    vR = gpuArray.zeros(N, n_nei, 'single');
    pL = gpuArray.zeros(N, n_nei, 'single');
    pR = gpuArray.zeros(N, n_nei, 'single');

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

    %% -------------------- 8) Y方向 Rusanov 通量 --------------------
    mask_y = ny >= 0;
    rL_y = gpuArray.zeros(N, n_nei, 'single');
    rR_y = gpuArray.zeros(N, n_nei, 'single');
    uL_y = gpuArray.zeros(N, n_nei, 'single');
    uR_y = gpuArray.zeros(N, n_nei, 'single');
    vL_y = gpuArray.zeros(N, n_nei, 'single');
    vR_y = gpuArray.zeros(N, n_nei, 'single');
    pL_y = gpuArray.zeros(N, n_nei, 'single');
    pR_y = gpuArray.zeros(N, n_nei, 'single');

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

    %% -------------------- 9) 中心通量 --------------------
    Hc = (gamma .* p11) ./ ((gamma-1) .* r) + 0.5 .* (u.^2 + v.^2);
    f_center = [r.*u, r.*u.*u + p11, r.*u.*v, r.*u.*Hc];
    g_center = [r.*v, r.*u.*v, r.*v.*v + p11, r.*v.*Hc];

    %% -------------------- 10) 残差 --------------------
    Dhx_face = Dhx_half(:,2:end);
    Dhy_face = Dhy_half(:,2:end);
    Dhx1 = Dhx_half(:,1);
    Dhy1 = Dhy_half(:,1);

    flux_x_sum = reshape(sum(Dhx_face .* flux_x, 2), [N,4]);
    flux_y_sum = reshape(sum(Dhy_face .* flux_y, 2), [N,4]);
    center_term = Dhx1 .* f_center + Dhy1 .* g_center;
    Res = flux_x_sum + flux_y_sum + center_term;

    %% -------------------- 11) 局部时间步长 --------------------
    % u_neighbors = u(nearest_indices(:,1:end));
    % v_neighbors = v(nearest_indices(:,1:end));
    % conv = abs(sum(Dhx_half .* u_neighbors + Dhy_half .* v_neighbors, 2));
    % geom = sqrt(Dhx_half.^2 + Dhy_half.^2);
    % sum_geom = sum(geom, 2);
    % acous = c .* sum_geom;
    % dt_local = CFL ./ max(conv + acous, eps_phys);
    dt_used = dt;      % 逐点时间步长，返回标量仅用于输出

    %% -------------------- 12) 雅可比近似 --------------------
    ax = sum(abs(Dhx_half), 2);
    ay = sum(abs(Dhy_half), 2);
    beta = ax .* (abs(u) + c) + ay .* (abs(v) + c);

    [Ablock, Bblock] = localEulerJacobiansGPU_safe(r,u,v,H,gamma);

    ax3    = reshape(ax,      1, 1, N);
    ay3    = reshape(ay,      1, 1, N);
    alpha3 = reshape(1 + dt_used .* beta, 1, 1, N);

    LHS = bsxfun(@times, dt_used, Ablock .* ax3 + Bblock .* ay3);
    LHS(1,1,:) = LHS(1,1,:) + alpha3;
    LHS(2,2,:) = LHS(2,2,:) + alpha3;
    LHS(3,3,:) = LHS(3,3,:) + alpha3;
    LHS(4,4,:) = LHS(4,4,:) + alpha3;

    %% -------------------- 13) 求解更新量 --------------------
    rhs = -bsxfun(@times, dt_used, Res);
    rhs3 = reshape(rhs.', [4,1,N]);
    dU3 = pagefun(@mldivide, LHS, rhs3);
    dU  = reshape(permute(dU3,[3,1,2]), [N,4]);

    Ua = Ua1 + dU;

    Ua(:,1) = max(Ua(:,1), eps_phys);
    rho_new = Ua(:,1);
    u_new = Ua(:,2) ./ rho_new;
    v_new = Ua(:,3) ./ rho_new;
    Emin = 0.5 .* rho_new .* (u_new.^2 + v_new.^2) + eps_phys/(gamma-1);
    Ua(:,4) = max(Ua(:,4), Emin);

    q = max(abs(dU(:,1)));
end

function [Ablock, Bblock] = localEulerJacobiansGPU_safe(r, u, v, H, gamma)
    N = numel(r);
    q2 = u.^2 + v.^2;
    Ablock = gpuArray.zeros(4, 4, N, 'single');
    Bblock = gpuArray.zeros(4, 4, N, 'single');

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
 
