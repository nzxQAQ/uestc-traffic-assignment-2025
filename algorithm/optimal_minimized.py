# 简单的二分法实现（未使用）
# def find_optimal_alpha(obj, bounds, tol=1e-3, max_iter=50):
    
#     a, b = bounds
#     for _ in range(max_iter):
#         mid = (a + b) / 2
#         f_mid = obj(mid)
#         f_a = obj(a)
#         f_b = obj(b)
        
#         # 检查收敛
#         if abs(b - a) < tol:
#             break
        
#         # 更新搜索范围
#         if f_mid < f_a and f_mid < f_b:
#             # 如果 mid 是最小值，缩小范围
#             if obj(mid - tol) < obj(mid + tol):
#                 b = mid
#             else:
#                 a = mid
#         elif f_a < f_b:
#             b = mid
#         else:
#             a = mid
    
#     return (a + b) / 2

# 使用 SciPy 库的 minimize_scalar 方法（未使用）
from scipy.optimize import minimize_scalar

def find_optimal_alpha(obj, bounds=(0, 1)):
    res = minimize_scalar(obj, bounds=bounds, method='bounded')
    return res.x

# 黄金分割法实现
# def find_optimal_alpha(obj, bounds=(0, 1), tol=1e-2, max_iter=50):
#     a, b = bounds
#     phi = (1 + 5**0.5) / 2  # 黄金比例
#     c = b - (b - a) / phi
#     d = a + (b - a) / phi
#     fc = obj(c)
#     fd = obj(d)

#     for _ in range(max_iter):
#         if abs(b - a) < tol:
#             break
#         if fc < fd:
#             b, fd = d, fc
#             d = c
#             c = b - (b - a) / phi
#             fc = obj(c)
#         else:
#             a, fc = c, fd
#             c = d
#             d = a + (b - a) / phi
#             fd = obj(d)

#     return (a + b) / 2

if __name__ == '__main__':
    # for test
    def obj(alpha):
        return alpha**2 + 2*alpha + 1
    
    bounds = (0, 1)
    alpha = find_optimal_alpha(obj, bounds)
    print(f'The optimal alpha is {alpha:.4f}')