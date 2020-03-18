function wider_result(model_name, set_list,dir_ext,seting_class,dateset_class)
fprintf('==========================================\n')
method_list = dir(dir_ext);
model_num = size(method_list,1) - 2;
candidate_model_name = cell(model_num,1);

for i = 3:size(method_list,1)
    candidate_model_name{i-2} = method_list(i).name;
end

for i = 1:size(set_list,1)
    propose = cell(model_num,1);
    recall = cell(model_num,1);
    name_list = cell(model_num,1);
    ap_list = zeros(model_num,1);
    for j = 1:model_num
        if strcmp(model_name, candidate_model_name{j}) == 1
            load(sprintf('%s/%s/wider_pr_info_%s_%s.mat',dir_ext, model_name, model_name, set_list{i}));
            propose{j} = pr_cruve(:,2);
            recall{j} = pr_cruve(:,1);
            ap = VOCap(propose{j},recall{j});
            ap_list(j) = ap;
            ap = num2str(ap);
            fprintf( '%s AP of %s is %s\n', legend_name, set_list{i} , ap)
        end
    end
end

fprintf('==========================================\n')

