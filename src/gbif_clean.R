# GBIF 기록 좌표 정제 표시 (CoordinateCleaner; Zizka et al. 2019, Methods Ecol Evol 10:744-751)
#
# 기록을 지우지 않고 검사별로 통과(TRUE)/걸림(FALSE)만 표시한다. 무엇을 제외할지는 다음 단계에서 정한다.
# 검사 목록:
#   - clean_coordinates()의 기본 검사(패키지 문서 기본값): capitals, centroids, equal, gbif,
#     institutions, outliers, seas, zeros
#   - 추가: urban (SAGE 벤치마크, arXiv 2609.31082가 사용), duplicates (같은 종·같은 좌표)
# 각 검사의 세부 기준값은 모두 패키지 기본값을 그대로 쓴다(바꾸지 않음).
#
# 입력: data/gbif/occurrences_flagged.parquet
# 출력: data/gbif/cc_flags.parquet (gbifID + 검사별 결과), data/gbif/cc_session.txt (버전 기록)
# 실행: Rscript src/gbif_clean.R   (프로젝트 루트에서)

suppressPackageStartupMessages({
  library(arrow)
  library(CoordinateCleaner)
})

occ <- read_parquet("data/gbif/occurrences_flagged.parquet",
                    col_select = c("gbifID", "count_key", "decimalLongitude", "decimalLatitude"))
occ <- as.data.frame(occ)
occ$count_key <- as.character(occ$count_key)

tests <- c("capitals", "centroids", "equal", "gbif", "institutions", "outliers",
           "seas", "zeros", "urban", "duplicates")

flags <- clean_coordinates(
  x = occ,
  lon = "decimalLongitude",
  lat = "decimalLatitude",
  species = "count_key",
  tests = tests,
  value = "spatialvalid"
)

out <- data.frame(gbifID = occ$gbifID, flags[, grepl("^\\.", names(flags))])
write_parquet(out, "data/gbif/cc_flags.parquet")

sink("data/gbif/cc_session.txt")
cat("run_at:", format(Sys.time(), tz = "UTC", usetz = TRUE), "\n")
cat("tests:", paste(tests, collapse = ", "), "\n\n")
print(sessionInfo())
sink()

print(colSums(!out[, -1], na.rm = TRUE))
