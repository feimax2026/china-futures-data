# 商品研究平台 v1（2026-10）

## 范围与仓库边界

`china-futures-data` 负责采集、数据质量、研究预测与历史版本；
`commodity-sentinel` 仍负责事件与日报消费。`JM_QUANT` 和 `my-futures-db`
保留历史，不迁移或删除。

期货：JM、I、SM、CU、AU、AG、SC、AL、SI、SF。
期权：AU、AG、CU、JM，另保留原油 SC。
AL/SI/SF/SM 是优先研究电力成本暴露的品种；不能把焦煤当成动力煤，
也不能把原油价格当成中国冶炼企业电费的直接代理。

## 数据到底存在哪里

Google Cloud 项目：`market-brief-wfm-20260525`。
专用私有桶：`gs://market-brief-wfm-20260525-futures-research`，东京。
这是 Google Cloud Storage，不是 Firestore 数据库；不使用 Firebase 客户端规则。
本地镜像为 `data/lake/`；CSV 工作缓存为 `data/raw/`，DuckDB 为 `data/db/`。
不把大量行情、账户凭证、预测历史塞进 Git。

```
raw/                      原始交易所 JSON + 来源/抓取时间/哈希
snapshots/main_observed/  历史行情源每次观察的 Parquet 快照
snapshots/futures_chain/  实际挂牌月份合约的日行情
snapshots/options_chain/  期权链、同月标的映射、质量标记
snapshots/oos_5d/         每周样本外比较（10d 同理）
snapshots/power_external/ 带发布时间的外部电力数据（需另行接入）
forecasts/legacy/         当时实际发出的四品种预测（本地/归档）
forecasts/research/       当时实际生成的扩展研究预测及错误状态
models/                  按品种/周锚点冻结的模型 JSON（不用 pickle）
state/main/              可更新的历史工作缓存
state/research/latest.json 最新研究输出（可能 partial，不是交易指令）
```

桶禁止公共访问、启用统一 IAM 与对象版本历史。研究服务账号只能读及新建
历史对象，只在 `state/` 前缀有覆盖权限。GitHub 使用 OIDC，无长期服务账号密钥。
存储和版本历史按 GCS 实际用量收费，未设置会自动删除研究历史的生命周期规则。

## 自动更新与兼容性

旧 `Update China Futures Data` 工作流仍按原时间运行，`latest_signals.json`
保持 schema=1 和 JM/I/SM/CU 四品种，不破坏现有日报。
新 `Collect Expanded Commodity Research` 独立采集全品种，计划东京 03:35；
每周一另跑近一年、按周重训的 baseline 样本外评估。GitHub 定时可能延迟。
新工作流合入 main 后定时才生效；开发分支可手动运行验证。

新增数据失败不会改写旧四品种 JSON，也不会把缺失品种填成最新行情。
研究报告保留错误并标为 `partial`，工作流返回失败以提醒维护。
节假日 `expected_source_date` 为前一个中国交易日，不要求当天出现行情。
本地命令：

```sh
python src/research_pipeline.py
python src/research_pipeline.py --evaluate
python src/research_pipeline.py --trade-date 2026-09-30 --chains-only
python src/build_duckdb.py
python src/import_power_data.py your_power_data.csv
```

本地未配置 `RESEARCH_BUCKET` 时只保存到本地，不上传。
配置桶时还必须有 Google Application Default Credentials；GitHub 工作流由 OIDC 提供。
`--allow-partial` 只用于本地诊断，生产不使用。
历史链按交易日手动补采，不把“今天看到的合约列表”当成过去的挂牌列表。

## 研究纪律

主力连续保留为旧系统兼容与初步研究输入，不是可执行交易收益。
已有持仓加权指数和真实合约历史下载工具保留；新链快照提供可追溯的后续数据。
下一阶段必须检查逐合约的相邻交易日、完整性和换月成本后，再推广加权指数研究。

预测 5/10 个交易日对数收益，比较 zero / momentum / 标准化 Ridge / XGBoost。
新研究每周首个交易日为重训锚点，同周每日更新特征；训练标签的
`label_end_date` 必须严格早于重训锚点，样本外比较使用同样的周节奏。
历史滚动评估不是历史上真实发布的预测；两类归档分别保存。
同一周模型权重冻结，后续运行从 GCS/本地复用；记录实际 `model_trained_at`，
不会把首次补建模型的抓取/训练时间伪装成过去。旧四品种兼容模型仍保持原训练节奏。
历史源被修订时，每次观察都另存，不声称回填数据是当时已知数据。
保留实际产生的预测时间，不以今天重新训练的回测替换当时预测。
重叠 5/10 日标签不独立，准确率不能直接解读为可交易的优势或利润。
波动率 v1 只输出 20d 历史与 EWMA 年化基线，未来 5/20d HAR 等模型待验证。

Sina 历史结算价 0 标记为缺失，不填充收盘价。异常历史 OHLC 保留原始快照、
清洗层标记无效且不删除日期，避免删行后把“5日”变短。
最新交易日价格/结算缺失、过期或特征不足则拒绝预测。

## 期权边界

精确解析同月份期货、看涨/看跌、行权价、合约乘数与最小跳动；
不使用 AU0/JM0 连续价作为期权标的。保留零成交合约但不视为可交易报价。
交易所 Delta/IV 与自行估算分列，缺少数据不插值补成完整曲面。
到期日尚未接入已核验的逐合约日历，`expiry_verified=false`；因此 v1
不会生成自算 IV 曲面、Greeks 或卖期权建议。美式期权不能直接套欧式定价。
黄金/铜旧月份存在欧式到美式过渡，元数据按合约月份区分。

焦煤期权 2026 年才上市，不能伪造多年历史。
AKShare 内置大商所期权列表未包括 JM，适配器直接请求大商所当前 JSON 接口。
交易所地域限制/反爬/超时会明确报告失败，不绕过访问控制、不替换成其他品种。

规则来源：
- https://www.shfe.com.cn/services/indexopt/contractrules/
- https://tsite.shfe.com.cn/upload/20201113/1605238029718.pdf
- https://www.ine.com.cn/eng/market/options/sc/contract/
- https://www.dce.com.cn/dce/file/2026-01-15/17684624156122c9a882b9ae6dcbb289019bc092f6fc1681.pdf

## 电价研究：有接口，不假装已有数据

外部 CSV 必須含 region / metric / delivery_date / published_at / available_at /
value / unit / source_url。所有发布时间有明确时区；特征只能使用决策时点前
`available_at` 已到的数据。允许负电价，不自动把负价格作为错误删除。
分开日前/实时现货价、工业电价、冶炼企业实际电费、水电与负荷。
实际省级、企业级电价及订阅数据源仍需确定；目前未连接，不加入生产模型。
优先云南/四川水电—铝/硅，内蒙古/宁夏电力成本—硅铁/锰硅；
形成产业链假设后分别评估解释力与交易收益，避免将相关性当成因果。

## 后续验收

1. 焦煤期权接口在生产运行地点验证成功，补齐 2026 年有效历史。
2. 接入官方逐合约到期日与规则版本，确认可用报价后实现美式 IV/Greeks。
3. 逐合约连续性/缺口检查与真实换月执行回测，不能凭主力价格曲线宣称收益。
4. 实际电力数据源接入，按历史可获得时间连接，再加入波动率/跨品种模型。
5. 新模型须稳定优于基线、经过含费用样本外检验，才进入候选交易信号。
