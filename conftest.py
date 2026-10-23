"""Gom bytecode (.pyc) khi chạy pytest vào MỘT thư mục `__pycache__` ở gốc ai-service,
thay vì một `__pycache__` cạnh mỗi thư mục mã nguồn.

Phải nằm ở gốc: chính file này được biên dịch trước khi dòng dưới chạy, nên pyc của nó rơi vào
`__pycache__` ở gốc, đúng thư mục dùng chung. (tests/conftest.py sẽ để lại tests/__pycache__.)
Bỏ qua khi đã đặt PYTHONPYCACHEPREFIX hoặc PYTHONDONTWRITEBYTECODE (vd trong Docker).
"""
import os
import sys

if not sys.pycache_prefix and not sys.dont_write_bytecode:
    sys.pycache_prefix = os.path.join(os.path.dirname(os.path.abspath(__file__)), "__pycache__")
