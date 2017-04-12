//==============================================================================
//	
//	Copyright (c) 2017-
//	Authors:
//	* Dave Parker <d.a.parker@cs.bham.ac.uk> (University of Birmingham)
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

package demos;

import java.io.FileOutputStream;
import java.io.PrintStream;


import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;

import parser.ast.*;
import parser.*;
import parser.visitor.*;
import prism.PrismLangException;
import prism.PrismUtils;
import parser.type.*;

import prism.ModelType;
import prism.Prism;
import prism.PrismDevNullLog;
import prism.PrismException;
import prism.PrismLog;
import prism.Result;
import prism.UndefinedConstants;


import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;


/**
 * An example class demonstrating how to control PRISM programmatically,
 * through the functions exposed by the class prism.Prism.
 * 
 * This shows how to load a model from a file and model check some properties,
 * either from a file or specified as a string, and possibly involving constants. 
 * 
 * See the README for how to link this to PRISM.
*/ 
public class grid_world
{

	public static void main(String[] args) throws IOException, InterruptedException,ParseException, PrismLangException
	{
		//Process proc = Runtime.getRuntime().exec("python /Users/weichaozhou/Documents/Safe_AI_MDP/workspace/grid_world/cirl/run.py");  
		//proc.waitFor();  
		new grid_world().run();
	}
	
	static final public void ConstantDef(ConstantList constantList, ArrayList<String> lines) {
		String sLastLine = lines.get(0), sCurrentLine = lines.get(1);
		for(String line: lines) {
			if(lines.indexOf(line) % 2 == 1) {
				sCurrentLine = line;
				try {
					//Integer.parseInt(sCurrentLine);
					constantList.addConstant(new ExpressionIdent(sLastLine), new ExpressionLiteral(TypeInt.getInstance(), Integer.parseInt(sCurrentLine)), TypeInt.getInstance());
				}	catch (NumberFormatException e) {
					constantList.addConstant(new ExpressionIdent(sLastLine), new ExpressionLiteral(TypeDouble.getInstance(), Double.parseDouble(sCurrentLine)), TypeDouble.getInstance());
				}
			}
			else {
				sLastLine = line;
			}
		}
		constantList.addConstant(new ExpressionIdent("e"), new ExpressionLiteral(TypeDouble.getInstance(), 2.72), TypeDouble.getInstance());
		constantList.addConstant(new ExpressionIdent("x_min"), new ExpressionLiteral(TypeInt.getInstance(), 0), TypeInt.getInstance());
		constantList.addConstant(new ExpressionIdent("y_min"), new ExpressionLiteral(TypeInt.getInstance(), 0), TypeInt.getInstance());
		constantList.addConstant(new ExpressionIdent("x_init"), new ExpressionLiteral(TypeInt.getInstance(), 0), TypeInt.getInstance());
		constantList.addConstant(new ExpressionIdent("y_init"), new ExpressionLiteral(TypeInt.getInstance(), 0), TypeInt.getInstance());
		
		//System.out.println(constantList);
	}
	
	static final public void Prop_ConstantDef(ConstantList constantList,int x_init, int y_init, int x_end, int y_end) {
//		constantList.addConstant(new ExpressionIdent("x_init"), new ExpressionLiteral(TypeInt.getInstance(), x_init), TypeInt.getInstance());
//		constantList.addConstant(new ExpressionIdent("y_init"), new ExpressionLiteral(TypeInt.getInstance(), y_init), TypeInt.getInstance());
		constantList.addConstant(new ExpressionIdent("x_end"), new ExpressionLiteral(TypeInt.getInstance(), x_end), TypeInt.getInstance());
		constantList.addConstant(new ExpressionIdent("y_end"), new ExpressionLiteral(TypeInt.getInstance(), y_end), TypeInt.getInstance());

	}
	
	static final public void Prop_PropertyDef(PropertiesFile pf_expert) {
		String property1 = "Pmin=?[F (x=x_end & y=y_end)]";
		pf_expert.addProperty(new Property(new ExpressionLiteral(TypeDouble.getInstance(), property1)));
	}
	
	static final public void FormulaDef(FormulaList formulaList, ArrayList<String> lines) {
		 String stay = new String("("), right = new String("("), down = new String("("), left = new String("("), up = new String("("), sink = new String("(");
		 for(int y = 0; y < lines.size(); y++) { 
			 String[] actions = lines.get(y).split(":");
			 for(int x = 0; x < actions.length; x++)	{
				 switch((int)Double.parseDouble(actions[x]))	{
			      case 0:	stay = build_expr(stay, x, y);	break;
			      case 1:	right = build_expr(right, x, y);	break;
			      case 2:	down = build_expr(down, x, y);	break;
			      case 3:	left = build_expr(left, x, y);	break;
			      case 4:	up = build_expr(up, x, y);	break;
			      default:	sink = build_expr(sink, x, y);
			        break;
			     }
			 }
		 }
		 
		 stay = stay + ")";
		 right = right + ")";
		 down = down + ")";
		 left = left + ")";
		 up = up + ")";
		 sink = sink + ")";
		 
		 Expression stay_expr = new ExpressionLiteral(TypeBool.getInstance(), stay);
		 Expression right_expr = new ExpressionLiteral(TypeBool.getInstance(), right);
		 Expression down_expr = new ExpressionLiteral(TypeBool.getInstance(), down);
		 Expression left_expr = new ExpressionLiteral(TypeBool.getInstance(), left);
		 Expression up_expr = new ExpressionLiteral(TypeBool.getInstance(), up);
		 //Expression sink_expr = new ExpressionLiteral(TypeBool.getInstance(), sink);
		 
		 formulaList.addFormula(new ExpressionIdent("stay"), stay_expr);
		 formulaList.addFormula(new ExpressionIdent("right"), right_expr);
		 formulaList.addFormula(new ExpressionIdent("down"), down_expr);
		 formulaList.addFormula(new ExpressionIdent("left"), left_expr);
		 formulaList.addFormula(new ExpressionIdent("up"), up_expr);
		 //formulaList.addFormula(new ExpressionIdent("sink"), sink_expr);
		 //System.out.println(formulaList);
	 }
		 
	static final public String build_expr(String action, int x, int y) {
		if(action.equals("(")) {
			action = action + "(x=" + String.valueOf(x) + " & y=" + String.valueOf(y) + ")";  
		} else {
			action = action + " | (x=" + String.valueOf(x) + " & y=" + String.valueOf(y) + ")";  
		}
		return action;
	}
	
	static final public Module Module(ConstantList constantList, FormulaList formulaList) {
		Module m = new Module("grid_world");
		m.setName("grid_world");
		m.addDeclaration(new Declaration("x", new DeclarationInt(new ExpressionLiteral(TypeInt.getInstance(), 0), constantList.getConstant(constantList.getConstantIndex("x_max")))));
		m.addDeclaration(new Declaration("y", new DeclarationInt(new ExpressionLiteral(TypeInt.getInstance(), 0), constantList.getConstant(constantList.getConstantIndex("y_max")))));
		build_cmd(m, constantList, formulaList);
		return m;
	}
	
	static final public void build_cmd(Module m, ConstantList constantList, FormulaList formulaList) {
		for(int i = 1; i < formulaList.size(); i++) {
			Command c = new Command();
			Updates us = new Updates();
			Update u = new Update();
			c.setSynch(formulaList.getFormulaName(i));
			c.setSynchIndex(i);
			c.setGuard(new ExpressionLiteral(TypeBool.getInstance(), formulaList.getFormulaNameIdent(i)+"=true"));
			
			u.addElement(new ExpressionIdent("x"), new ExpressionLiteral(TypeInt.getInstance(), "x"));
			u.addElement(new ExpressionIdent("y"), new ExpressionLiteral(TypeInt.getInstance(), "y"));;
			us.addUpdate(new ExpressionLiteral(TypeDouble.getInstance(), "(1-p)/4"), u);
			u = new Update();
			u.addElement(new ExpressionIdent("x"), new ExpressionLiteral(TypeInt.getInstance(), "(x+1>x_max?x-1:x+1)"));
			u.addElement(new ExpressionIdent("y"), new ExpressionLiteral(TypeInt.getInstance(), "y"));;
			us.addUpdate(new ExpressionLiteral(TypeDouble.getInstance(), "(1-p)/4"), u);
			u = new Update();
			u.addElement(new ExpressionIdent("x"), new ExpressionLiteral(TypeInt.getInstance(), "x"));
			u.addElement(new ExpressionIdent("y"), new ExpressionLiteral(TypeInt.getInstance(), "(y+1>y_max?y-1:y+1)"));;
			us.addUpdate(new ExpressionLiteral(TypeDouble.getInstance(), "(1-p)/4"), u);
			u = new Update();
			u.addElement(new ExpressionIdent("x"), new ExpressionLiteral(TypeInt.getInstance(), "(x-1<x_min?x+1:x-1)"));
			u.addElement(new ExpressionIdent("y"), new ExpressionLiteral(TypeInt.getInstance(), "y"));;
			us.addUpdate(new ExpressionLiteral(TypeDouble.getInstance(), "(1-p)/4"), u);
			u = new Update();
			u.addElement(new ExpressionIdent("x"), new ExpressionLiteral(TypeInt.getInstance(), "x"));
			u.addElement(new ExpressionIdent("y"), new ExpressionLiteral(TypeInt.getInstance(), "(y-1<y_min?y+1:y-1)"));;
			us.addUpdate(new ExpressionLiteral(TypeDouble.getInstance(), "(1-p)/4"), u);
			
			us.setProbability(i, new ExpressionLiteral(TypeDouble.getInstance(), "p"));
			u = new Update();
			c.setUpdates(us);
			m.addCommand(c);
		}
	/**	Command c = new Command();
		Updates us = new Updates();
		Update u = new Update();
		c.setSynch(formulaList.getFormulaName(formulaList.size()-1));
		c.setSynchIndex(formulaList.size()-1);
		c.setGuard(formulaList.getFormulaNameIdent(formulaList.size()-1));
		u.addElement(new ExpressionIdent("x"), new ExpressionLiteral(TypeInt.getInstance(), "x"));
		u.addElement(new ExpressionIdent("y"), new ExpressionLiteral(TypeInt.getInstance(), "y"));;
		us.addUpdate(new ExpressionLiteral(TypeDouble.getInstance(), "1"), u);
		m.addCommand(c);**/
	}
	
	static final public void run() throws ParseException, InterruptedException, FileNotFoundException	{
		try {
			// Create a log for PRISM output (hidden or stdout)
			PrismLog mainLog = new PrismDevNullLog();
			//PrismLog mainLog = new PrismFileLog("stdout");
			
			// Initialise PRISM engine 
			Prism prism = new Prism(mainLog); 
			prism.initialise(); 
			
			ModulesFile mf_expert = new ModulesFile();
			ModulesFile mf_demo = new ModulesFile();
			
			mf_expert.setModelType(ModelType.DTMC);
			mf_demo.setModelType(mf_expert.getModelType());
			
			ArrayList<String> files = new ArrayList<String>();
			String STATE_SPACE = "/home/zwc662/Documents/prism-4.3.1-src/src/demos/state_space";
			String EXPERT_POLICY = "/home/zwc662/Documents/prism-4.3.1-src/src/demos/expert_policy";
			String DEMO_POLICY = "/home/zwc662/Documents/prism-4.3.1-src/src/demos/demo_policy";
			files.add(STATE_SPACE);
			files.add(EXPERT_POLICY);
			files.add(DEMO_POLICY);
			ArrayList<String> lines = new ArrayList<String>();
			for(String file: files) {
				BufferedReader br = null;
				FileReader fr = null;
				try {
					fr = new FileReader(file);
					br = new BufferedReader(fr);
					String line;
					br = new BufferedReader(new FileReader(file));
					while ((line = br.readLine()) != null) {
						lines.add(line);
					}
					if(file.equals(STATE_SPACE)) {	
						ConstantDef(mf_expert.getConstantList(), lines);	
						mf_demo.setConstantList(mf_expert.getConstantList());
						lines.clear();
						}
					if(file.equals(EXPERT_POLICY)) {	
						FormulaDef(mf_expert.getFormulaList(), lines);	
						lines.clear();
						}
					if(file.equals(DEMO_POLICY)) {	
						FormulaDef(mf_demo.getFormulaList(), lines);	
						lines.clear();
						}
				} catch (IOException e) {	
					e.printStackTrace();
				}
			}
			

			Module m_expert = Module(mf_expert.getConstantList(), mf_expert.getFormulaList());
            mf_expert.addModule(m_expert);
            mf_expert.setInitialStates(new ExpressionLiteral(TypeBool.getInstance(), "x = 5 & y = 5"));
            
            mf_expert.tidyUp();
            Module m_demo = Module(mf_demo.getConstantList(), mf_expert.getFormulaList());
            mf_demo.addModule(m_demo);
            mf_demo.setInitialStates(new ExpressionLiteral(TypeBool.getInstance(), "x =5 & y = 4"));
            mf_demo.tidyUp();
            
            
            //RewardStruct rs = RewardStruct();
            //mf.addRewardStruct(rs);
            
            //Expression init = Init();
            //mf.setInitialStates(init); initCount++; if (initCount == 2) initDupe = init;
           
            prism.loadPRISMModel(mf_expert);
            
            PrintStream ps_console = System.out;
	        PrintStream ps_file = new PrintStream(new FileOutputStream(new File("/home/zwc662/Documents/prism-4.3.1-src/src/demos/grid_world.pm")));
	        System.setOut(ps_file);
			System.out.println(mf_expert);
			
			System.setOut(ps_console);
	        System.out.println(mf_expert);
			
	        ModulesFile modulesFile = prism.parseModelFile(new File("/home/zwc662/Documents/prism-4.3.1-src/src/demos/grid_world.pm"));
			prism.loadPRISMModel(modulesFile);
			// Parse and load a properties model for the model
			PropertiesFile propertiesFile = prism.parsePropertiesFile(modulesFile, new File("/home/zwc662/Documents/prism-4.3.1-src/src/demos/grid_world.pctl"));
			Result result = prism.modelCheck(propertiesFile, propertiesFile.getPropertyObject(0));
			System.out.println(result.getResult());
			//System.out.println(mf_demo);
			//PropertiesFile pf_expert =  new PropertiesFile(mf_expert);
			//Prop_ConstantDef(propertiesFile.getConstantList(), 0, 0, 1, 1);
			//System.out.println(pf_expert.getConstantList());
			
			propertiesFile = prism.parsePropertiesString(mf_expert, "filter(min, P=? [F x = 4 & y = 5], (x!=4|y!=5)&(x!=2|y!=2)&(x!=5|y!=2))");	
			result = prism.modelCheck(propertiesFile, propertiesFile.getPropertyObject(0));
			System.out.println(result.getResult());
			
			
			
			
			/**
			List<String> consts = pf_expert.getUndefinedConstants();
			if(consts.isEmpty() != true) {
				for(int i = 0; i<7; i++) {
					for(int j = 0; j<7; j++) {
						ArrayList<Object> values = new ArrayList<Object>();
						values.add(i);
						values.add(j);
						Values Values = new Values();
						//System.out.println(pf_expert.getUndefinedConstants());
						Values.setValue(pf_expert.getUndefinedConstants().get(0), i);
						Values.setValue(pf_expert.getUndefinedConstants().get(1), j);
						//System.out.println(Values);
						pf_expert.setUndefinedConstants(Values);
						
					}
				}
				
			}

			// Model check the second property from the file
			System.out.println("which has an undefined constant, which we check over a range 0,1,2");
			UndefinedConstants undefConsts = new UndefinedConstants(modulesFile, propertiesFile, propertiesFile.getPropertyObject(1));
			undefConsts.defineUsingConstSwitch(constName + "=0:2");
			int n = undefConsts.getNumPropertyIterations();
			for (int i = 0; i < n; i++) {
				Values valsExpt = undefConsts.getPFConstantValues();
				propertiesFile.setUndefinedConstants(valsExpt);
				System.out.println(propertiesFile.getPropertyObject(1) + " for " + valsExpt);
				result = prism.modelCheck(propertiesFile, propertiesFile.getPropertyObject(1));
				System.out.println(result.getResult());
				undefConsts.iterateProperty();
			}

			// Model check a property specified as a string
			propertiesFile = prism.parsePropertiesString(modulesFile, "P=?[F<=5 s=7]");
			System.out.println(propertiesFile.getPropertyObject(0));
			result = prism.modelCheck(propertiesFile, propertiesFile.getPropertyObject(0));
			System.out.println(result.getResult());

			// Model check an additional property specified as a string
			
			result = prism.modelCheck(propertiesFile, propertiesFile.getPropertyObject(0));
			System.out.println(result.getResult());

			// Close down PRISM
			prism.closeDown();
			
		} catch (FileNotFoundException e) {
			System.out.println("Error: " + e.getMessage());
			System.exit(1);
**/		} catch (PrismException e) {
			System.out.println("Error: " + e.getMessage());
			System.exit(1);
		}
		
	}
}