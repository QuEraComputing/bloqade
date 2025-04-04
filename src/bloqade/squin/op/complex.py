# Stopgap Measure, squin dialect needs Complex type but
# this is only available in Kirin 0.15.x

from kirin.ir.attrs.types import PyClass

Complex = PyClass(complex)
