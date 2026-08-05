#
# This file is part of the CEMD distribution
# Copyright (c) 2022-2026 Jérôme Claverie.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

set env(VMDFORCECPUCOUNT) 4

package require pbctools
package require topotools

# --- PARSE COMMAND LINE ARGUMENTS ---

set file_topo [lindex $argv 0]
set file_trj ""
set config ""
set rep_file ""

for {set i 1} {$i < $argc} {incr i} {
    set arg [lindex $argv $i]

    switch -- $arg {
        "--config" {
            incr i
            set config [lindex $argv $i]
        }
        "--rep" {
            incr i
            set rep_file [lindex $argv $i]
        }
        default {
            if {$file_trj eq ""} {
                set file_trj $arg
            }
        }
    }
}

# Load optional configuration file
if {$config ne ""} {
    source $config
}

# --- READ TOPOLOGY ---

set ext_topo [string tolower [file extension $file_topo]]

if {$ext_topo eq ".data"} {
    set system [topo readlammpsdata $file_topo full]
} else {
    set system [mol new $file_topo waitfor all]
}


# --- APPLY ELEMENT MAP ---

if {[info exists element_map]} {
    foreach element [array names element_map] {
        set types $element_map($element)

        set selection "type [lindex $types 0]"
        foreach t [lrange $types 1 end] {
            append selection " or type $t"
        }

        set sel [atomselect $system $selection]
        $sel set element $element
        $sel delete
    }
}

# Remove previous representation
mol delrep 0 top
topo clearbonds

# Display and Box settings
display projection orthographic
display rendermode GLSL
display resetview
display depthcue off
pbc box -center com -color gray -style lines -width 0.5 -material AOEdgy
axes location off

# Color and texture settings
color Display Background white

mol color Element

if {$rep_file ne ""} {
    source $rep_file
}

# Function that can be called inside vmd
proc render_tachyon {{ofile "output.png"} {resx 2048} {resy 2048}} {

    render Tachyon tmp.dat tachyon -aasamples 12 %s -format PNG -res $resx $resy -o $ofile

}

# Move all the atoms by "x y z"
proc move_by {x y z} {

    set nframes [molinfo 0 get numframes]
    set all [atomselect top all]

    for {set frame 0} {$frame < $nframes} {incr frame} {
        animate goto $frame
        $all moveby "$x $y $z"
    }

    pbc wrap -all

}


