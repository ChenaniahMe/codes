function [fea_v] = f_tensor2f_vector(fea)

[m, n, d] = size(fea);
fea_v = zeros(d, m*n);

for i = 1:m
    for j = 1:n
        fea_v(:, (i-1)*n+j) = fea(i, j, :);
    end
end

end