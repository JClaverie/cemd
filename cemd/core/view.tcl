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

package require pbctools
package require topotools

# --- READ TOPOLOGY AND TRAJECTORY ---

set file_topo [lindex $argv 0]
set ext_topo [lindex [split $file_topo .] end]

# Case 1: Only one file provided (LAMMPS data file or lammpstrj alone)
if {$argc == 1} {
    if {$ext_topo == "data"} {
        set system [topo readlammpsdata $file_topo full]
    } else {
        set system [mol new $file_topo waitfor all]
    }
}

# Case 2: Two files provided (Topology + Trajectory)
if {$argc == 2} {
    set file_trj [lindex $argv 1]
    set ext_trj [lindex [split $file_trj .] end]

    if {$ext_topo == "data"} {
        # 1. Read the LAMMPS topology correctly with TopoTools
        set system [topo readlammpsdata $file_topo full]
        
        # 2. Append the trajectory file on top of the generated molecule ID
        if {$ext_trj == "dcd"} {
            mol addfile $file_trj type dcd waitfor all $system
        } else {
            mol addfile $file_trj type lammpstrj waitfor all $system
        }
    } else {
        # Historical fallback if you ever pass a PSF file
        set system [mol new $file_topo]
        if {$ext_trj == "dcd"} {
            mol addfile $file_trj waitfor all
        } else {
            mol addfile $file_trj type lammpstrj waitfor all
        }
    }
}

topo clearbonds

# Display and Box settings
mol delrep 0 top
display projection orthographic
display rendermode GLSL
display resetview
display depthcue off
pbc box -center com -color gray -style lines -width 0.5 -material AOEdgy
axes location off

# Add non-existing color categories
# create dummy molecule with one atom
set mol [mol new atoms 1]
set sel [atomselect $mol all]
# add items to color categories
$sel set name K
$sel set name X
$sel set name Y
$sel set name Z
# clean up
$sel delete
mol delete $mol

# Reset atom name for coloring for elements with two letters
set sel [atomselect top "type Ca Cw Ca_s Ca_aq"]
$sel set name X

set sel [atomselect top "type Cl"]
$sel set name Y

set sel [atomselect top "type Si"]
$sel set name Z

set sel [atomselect top "type Al"]
$sel set name A

# Color and texture settings
color Display Background white

mol material AOEdgy
material change outlinewidth AOEdgy 0.6
material change outline AOEdgy 2.0
material change shininess AOEdgy 0.8

color Name H white
color Name O red
color Name N violet
color Name K lime
color Name A purple

color Name X green3
color Name Y cyan3
color Name Z orange3

mol color Name

# Dynamic bonds of water and hydroxyl groups
mol selection type Ow Hw Ow_aq Hw_aq Ow_s Hw_s
mol representation DynamicBonds 1.2 0.200000 12.000000
mol addrep $system

mol selection type Oh Hh Oh_aq Hh_aq Oh_s Hh_s 
mol representation DynamicBonds 1.2 0.200000 12.000000
mol addrep $system

mol selection type Oah Ha Osih Hsi Oh H O Oh1 Hh1 
mol representation DynamicBonds 1.2 0.200000 12.000000
mol addrep $system

# Dynamic bonds in sulfates
mol selection type Os S O
mol representation DynamicBonds 1.6 0.200000 12.000000
mol addrep $system

# Dynamic bonds in carbonates
mol selection type Oc C O
mol representation DynamicBonds 1.6 0.200000 12.000000
mol addrep $system

# Dynamic bonds in silicates
mol selection type Osi Si
mol representation DynamicBonds 2 0.200000 12.000000
mol addrep $system

mol selection type Ob Si
mol representation DynamicBonds 2 0.200000 12.000000
mol addrep $system

mol selection type Obs Si
mol representation DynamicBonds 2 0.200000 12.000000
mol addrep $system

mol selection type Osih Si
mol representation DynamicBonds 2 0.200000 12.000000
mol addrep $system

mol selection type O Si
mol representation DynamicBonds 2 0.200000 12.000000
mol addrep $system

# Dynamic bonds in aluminates
mol selection type Oa Al
mol representation DynamicBonds 2 0.200000 12.000000
mol addrep $system

mol selection type Ob Al
mol representation DynamicBonds 2 0.200000 12.000000
mol addrep $system

mol selection type Obs Al
mol representation DynamicBonds 2 0.200000 12.000000
mol addrep $system

mol selection type Oh Al
mol representation DynamicBonds 2 0.200000 12.000000
mol addrep $system

mol selection type Oah Al
mol representation DynamicBonds 2 0.200000 12.000000
mol addrep $system

mol selection type O Al
mol representation DynamicBonds 2 0.200000 12.000000
mol addrep $system

mol selection type C H
mol representation DynamicBonds 1.2 0.200000 12.000000
mol addrep $system

# # Dynamic bonds between aluminates and silicates
# mol selection type Oa Si
# mol representation DynamicBonds 1.9 0.200000 12.000000
# mol addrep $system

# CPK drawing of oxygen and hydrogen in water
mol selection type Ow Hw Ow1 Hw1 O H
mol representation CPK 0.8 0.8 12.000000 12.000000
mol addrep $system

# CPK drawing of oxygen and hydrogen in hydroxides
mol selection type Hh Oh Hh1 Oh1 Oh_aq Hh_aq Oh_s Hh_s
mol representation CPK 0.8 0.8 12.000000 12.000000
mol addrep $system

# CPK drawing of oxygen and hydrogen in hydroxyles
mol selection type Ob Obs Osi Hsi Osih Oa Ha Oah
mol representation CPK 0.8 0.8 12.000000 12.000000
mol addrep $system

# CPK drawing of other atoms
mol selection type Cl Na Ca K Ca_s Ca_aq Cw
mol representation CPK 1.8 0.200000 12.000000 12.000000
mol addrep $system

mol selection type S Si C Al
mol representation CPK 1 0.8 12.000000 12.000000
mol addrep $system

mol selection type Os Oc
mol representation CPK 0.8 0.8 12.000000 12.000000
mol addrep $system


# Function that can be called inside vmd
proc screenshot {{box 1} {resx 4096} {resy 4096}} {

    if {$box == 0} {
        pbc box -off
    } 

    if {$box == 1} {
        pbc box -on
        pbc box -centersel all -color gray -style lines -width 0.5 -material AOEdgy
    } 
    
    set env(VMDFORCECPUCOUNT) 4
    render Tachyon tmp.dat tachyon -aasamples 12 %s -format PNG -res $resx $resy

}

# Move all the atoms by "x y z"
proc move {x y z} {

    set sel [atomselect top all]
    $sel moveby "$x $y $z"
    pbc wrap

}

# Write the current configuration as a LAMMPS datafile
proc write {file} {

    topo writelammpsdata $file

}

# Write the whole trajectory in the DCD format
proc writeall {file} {

    animate write dcd $file beg 0 end -1 waitfor all

}

# Move all the trajectory by the given vector
proc moveall {x y z} {

    set nframes [molinfo 0 get numframes]
    set all [atomselect top all]

    for {set frame 0} {$frame < $nframes} {incr frame} {
        animate goto $frame
        $all moveby "$x $y $z"
    }

    pbc wrap -all
}



