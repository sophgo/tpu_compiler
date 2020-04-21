from ..numpy_helper import npz_compare as base_npz_compare

class cvi_data(object):
    def __init__(self):
        pass

    @staticmethod
    def npz_compare(target_file: str, ref_file: str,
                verbose: int = 0,
                discard: int = 0,
                dtype: str=None,
                tolerance: str = "0.99,0.99,0.90",
                op_info: str = None,
                order: str=None,
                tensor: str=None,
                excepts: str = None,
                save: str=None,
                dequant: bool = False,
                full_array: bool = False,
                stats_int8_tensor: bool = False
                ):
        args = [
            target_file,
            ref_file,
            '--tolerance', tolerance,
        ]
        if dequant:
            args.append("--dequant")
        if full_array:
            args.append("--full-array")
        if stats_int8_tensor:
            args.append("--stats_int8_tensor")
        if verbose != 0:
            args_verbose = '-{}'.format('v'*verbose)
            args.append(args_verbose)
        if discard != 0:
            args_discard = '-{}'.format('d'*discard)
            args.append(args_discard)

        if dtype:
            args.append("--dtype")
            args.append(dtype)
        if op_info:
            args.append("--op_info")
            args.append(op_info)

        if order:
            args.append("--order")
            args.append(order)

        if tensor:
            args.append("--tensor")
            args.append(tensor)

        if excepts:
            args.append("--excepts")
            args.append(excepts)

        if save:
            args.append("--save")
            args.append(save)

        return base_npz_compare(args)