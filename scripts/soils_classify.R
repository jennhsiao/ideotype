### Classify soil texture for all NASS sites
# lack of good soil classification libraries in Python
# thus coding up in R

library('soiltexture')
library('dplyr')

df_soil <- read.csv('soils_nass.csv')
df_soil <- rename(df_soil, SAND = sand)
df_soil <- rename(df_soil, SILT = silt)
df_soil <- rename(df_soil, CLAY = clay)

df_soilstoclassify <- data.frame(
  'CLAY' = c(df_soil['CLAY']),
  'SILT' = c(df_soil['SILT']),
  'SAND' = c(df_soil['SAND'])
)

df_soilclass <-
  TT.points.in.classes(
    tri.data = df_soilstoclassify,
    class.sys = 'USDA.TT'
  )

# output as soils_nass_texture.csv
write.csv(df_soilclass, 'soils_nass_texture.csv')
