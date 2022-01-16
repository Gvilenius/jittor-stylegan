from jittor.utils.pytorch_converter import convert
pytorch_code = '''
out_std = math.sqrt(out.var(0, unbiased=False) + 1e-8)
'''

jittor_code = convert(pytorch_code)
print(jittor_code)
