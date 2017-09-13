//==============================================================================
//	
//	Copyright (c) 2002-
//	Authors:
//	* Andrew Hinton <ug60axh@cs.bham.ac.uk> (University of Birmingham)
//	* Dave Parker <david.parker@comlab.ox.ac.uk> (University of Oxford, formerly University of Birmingham)
//	
//------------------------------------------------------------------------------
//	
//	This file is part of PRISM.
//	
//	PRISM is free software; you can redistribute it and/or modify
//	it under the terms of the GNU General Public License as published by
//	the Free Software Foundation; either version 2 of the License, or
//	(at your option) any later version.
//	
//	PRISM is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//	GNU General Public License for more details.
//	
//	You should have received a copy of the GNU General Public License
//	along with PRISM; if not, write to the Free Software Foundation,
//	Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//	
//==============================================================================

package userinterface.model;
import java.awt.*;

// import java.util.Vector;
// import java.util.Properties;
// import javax.swing.text.*;
// import javax.swing.event.*;
// import java.util.*;
// import javax.swing.table.*;
// import javax.swing.*;
/**
 *
 * @author  ug60axh
 */
public class Style
{
    public Color c;
    public int style;
	
    public Style(Color c, int style)
    {
        this.c = c;
        this.style = style;
    }
    
    public static Style defaultStyle()
    {
        Style s = new Style(Color.black, Font.PLAIN);
        return s;
    }
    
    public Style copy()//grrrr
    {
        int r = c.getRed(), g = c.getGreen(), b = c.getBlue();
        Color cc = new Color(r,g,b);
        return new Style(cc, style);
    }
}