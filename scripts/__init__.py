"""ScaleSeek 的可执行脚本集合。

这个文件存在是必需的，不是可选的：没有它 `scripts/` 只是隐式命名空间包，
而命名空间包在 `sys.path` 扫描中优先级最低 —— vendored verl 里也有一个
`scripts/`（正规包），即使 ScaleSeek 在 PYTHONPATH 里更靠前，
`import scripts.run_official_baseline` 也会解析到 verl 的那个并失败。
"""
