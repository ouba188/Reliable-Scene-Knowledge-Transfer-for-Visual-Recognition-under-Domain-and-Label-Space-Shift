# 841 数据集 → 夸克网盘归档工具链（372993 · 2026-09-20）

服务器侧运行（文件在服务器上、带宽在服务器上）。凭据：`/root/quark_cookie.txt`（0600，浏览器登录态整串 Cookie），**不入库**。

## 用法

```bash
# 0) 清单：841 产品 -> (port, tif 路径, labels 路径)
python build_archive_inventory.py            # -> /root/ann841_inventory.tsv
# 1) 归档（tmux 内跑！）。P1 = tif + labels；P2 = png（本次先跳过，PNG 后补）
tmux new-session -d -s quark 'bash /root/run_quark_archive.sh'
# 想单独跑 P2：去掉 --skip-png 再执行 quark_archive.py，state 自动跳过已完成的
python quark_archive.py --uploaders 12 --converters 4 --part-mb 32 --skip-png
# 2) 核对（逐产品去重）
python count_ann_exact.py
```

网盘目录：`SAR_AIS_T_841_20260920/{tif,png,labels}/<港口>/<极化>/<文件>`

## 夸克上传协议（2026-09 实测，旧 `/file/upload/request` 已废弃）

```
1) POST /1/clouddrive/file/upload/pre      {file_name, format_type(mime), size, pdir_fid, l_created_at, ...}
                                           -> {task_id, upload_id, obj_key, bucket(ul-sz|ul-zb), upload_url(http://pds.quark.cn), auth_info, callback, metadata.part_size}
2) POST /1/clouddrive/file/update/hash     {md5, sha1, task_id} -> finish=true 即秒传（同账号已有同 md5 文件）
3) 每片: POST /1/clouddrive/file/upload/auth
      {auth_info, task_id, auth_meta}   # auth_meta = 手拼的 OSS 签名元数据（见 quark_upload.py auth_meta_put）
      -> data.auth_key  (作为 Authorization 头)
   PUT https://{bucket}.pds.quark.cn/{obj_key}?partNumber=N&uploadId=..  (Content-MD5 由 OSS 返回 etag)
4) POST 同 URL（x-oss-callback = base64(JSON(callback))）完成分片合并 -> {"Status":"OK"}
5) POST /1/clouddrive/file/upload/finish   {obj_key, task_id} -> 200
```

要点 / 坑：
- 主机必须是 `drive.quark.cn`（PC web；`/1/clouddrive/file/sort` 列目录用 **GET**，POST 会 405）。
- OSS 端点是 `{bucket}.pds.quark.cn` 的**公网域名**；`pds.quark.cn` 本身无公网 DNS（只在 ISP 网内），AutoDL 这类机房解析不了。
- `part_size` 服务端建议 4 MB，但是**可以由客户端自定**（本工具用 32 MB，减少 auth 往返）。
- 账号总带宽上限实测 ≈ **25 MB/s**（8→12 并发、4→32 MB 分片都不涨）；单流约 5 MB/s。
- **长任务必须 tmux 托管**：SSH 会话断开会杀子进程组（nohup/setsid 拦不住）；`pkill -f` 的模式别匹配到自己的 shell（`'[q]uark_archive.py'` 这种写法安全）。
- 全尺寸 PNG 实测只省 ~15% 体积（SAR 散斑熵高）且丢地理参考；要预览建议 1/4 缩略图。

协议完整版见本机 skill `quark-netdisk-upload`。
