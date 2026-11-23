-- IEEE two-column: small tables -> table/table* floats (tabular in adjustbox); big tables -> one-column longtable sandwich;
-- figures in the two-column body -> figure*. After the \appendices marker everything is one-column and left to pandoc.
local function tex(blocks) local s = pandoc.write(pandoc.Pandoc(blocks), 'latex'); return (s:gsub('%s+$', '')) end
local function cell(c) return tex(c.contents):gsub('\n\n', ' ') end
local function row_tex(r) local t = {} for _, c in ipairs(r.cells) do t[#t+1] = cell(c) end return table.concat(t, ' & ') .. ' \\\\' end
local function float_table(tbl, onecol)
  local ncol = #tbl.colspecs; local rows, chars = {}, 0
  for _, b in ipairs(tbl.bodies) do for _, r in ipairs(b.body) do local s = row_tex(r); rows[#rows+1] = s; chars = math.max(chars, #s) end end
  local head = {} for _, r in ipairs(tbl.head.rows) do head[#head+1] = row_tex(r) end
  local wide = (not onecol) and (ncol > 4 or chars > 90)
  local env = wide and 'table*' or 'table'; local width = (wide or onecol) and '\\textwidth' or '\\columnwidth'
  if onecol then env = 'table' end
  local spec = string.rep('l', ncol)
  local cap = tex(tbl.caption.long)
  local s = '\\begin{' .. env .. '}[!htbp]\\centering\\caption{' .. cap .. '}\\scriptsize\n\\begin{adjustbox}{max width=' .. width .. '}\n\\begin{tabular}{' .. spec .. '}\\toprule\n' ..
    table.concat(head, '\n') .. '\n\\midrule\n' .. table.concat(rows, '\n') .. '\n\\bottomrule\\end{tabular}\n\\end{adjustbox}\n\\end{' .. env .. '}'
  return pandoc.RawBlock('latex', s)
end
local function nrows(t) local n = 0 for _, bd in ipairs(t.bodies) do n = n + #bd.body end return n end
function Pandoc(doc)
  local out, appendix, deferred = {}, false, {}
  for _, b in ipairs(doc.blocks) do
    if b.t == 'RawBlock' and b.text:find('\\appendices', 1, true) then
      appendix = true; out[#out+1] = b
      if #deferred > 0 then
        out[#out+1] = pandoc.RawBlock('latex', '\\section{Long result tables}')
        for _, t in ipairs(deferred) do out[#out+1] = pandoc.RawBlock('latex', '{\\scriptsize\n' .. tex({t}) .. '\n}') end
      end
    elseif b.t == 'Table' then
      local longtext = false
      for _, bd in ipairs(b.bodies) do for _, r in ipairs(bd.body) do for _, c in ipairs(r.cells) do if #cell(c) > 70 then longtext = true end end end end
      if longtext and appendix then out[#out+1] = pandoc.RawBlock('latex', '{\\footnotesize\n' .. tex({b}) .. '\n}')
      elseif longtext then deferred[#deferred+1] = b
        out[#out+1] = pandoc.Para({pandoc.Emph({pandoc.Str('(Table in Appendix, "Long result tables".)')})})
      elseif nrows(b) <= 45 then out[#out+1] = float_table(b, appendix)
      elseif appendix then out[#out+1] = pandoc.RawBlock('latex', '{\\scriptsize\n' .. tex({b}) .. '\n}')
      else deferred[#deferred+1] = b
        out[#out+1] = pandoc.Para({pandoc.Emph({pandoc.Str('(Full table in Appendix, "Long result tables".)')})}) end
    elseif not appendix and b.t == 'Figure' then
      out[#out+1] = pandoc.RawBlock('latex', (tex({b}):gsub('\\begin{figure}', '\\begin{figure*}[!t]'):gsub('\\end{figure}', '\\end{figure*}')))
    else out[#out+1] = b end
  end
  doc.blocks = out; return doc
end
